"""The spine Scheduler's submit surface: outcomes, loop affinity, conversation
ids, and settlement under cancellation races."""

import asyncio
import contextlib

import pytest
from loguru import logger

from raven.spine import (
    BusyPolicy,
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from raven.spine.scheduler import _DEFAULT_IDLE_TTL, SchedulerDrainingError, _settle


def _req(
    *,
    conversation=None,
    channel="t",
    chat_id="c",
    origin=Origin.USER,
    busy=BusyPolicy.APPEND,
    text="hi",
    turn_id=None,
) -> TurnRequest:
    src = Source(channel=channel, chat_id=chat_id, sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(
        origin=origin, source=src, text=text, conversation=conversation, busy=busy, turn_id=turn_id
    )


async def _sink(event) -> None:
    pass


class SuccessRunner:
    def __init__(self, outcome=None):
        self.outcome = outcome or TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)

    async def run(self, req, emit, drain) -> TurnOutcome:
        return self.outcome


class HangingRunner:
    def __init__(self):
        self.started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.started.set()
        await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class FailingRunner:
    async def run(self, req, emit, drain) -> TurnOutcome:
        raise ValueError("boom")


class ConcurrentProbe:
    def __init__(self, target: int):
        self.running = 0
        self.reached = asyncio.Event()
        self._target = target

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.running += 1
        if self.running >= self._target:
            self.reached.set()
        await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def _scheduler(runner) -> Scheduler:
    return Scheduler(runner, OriginPools(user=1, system=1), _sink)


# --- submit / result / outcome channel ---


async def test_submit_result_returns_the_outcome():
    sched = _scheduler(SuccessRunner(TurnOutcome(usage=Usage(5, 7, 12), explicit_reply=True)))
    handle = sched.submit(_req())
    outcome = await handle.result()
    assert outcome == TurnOutcome(usage=Usage(5, 7, 12), explicit_reply=True)


async def test_failed_turn_result_is_the_report_the_lane_filed():
    # A failure and a cancel both used to arrive as None, so a submitter that
    # only awaits the handle -- cron -- had to invent a sentence for a failure
    # the turn had already worded.
    sched = _scheduler(FailingRunner())
    result = await sched.submit(_req()).result()
    assert isinstance(result, TurnFailed)
    assert (result.error, result.cancelled) == ("ValueError: boom", False)


# --- off-loop fail-fast (must compare self._loop, not just get_running_loop) ---


def test_submit_with_no_running_loop_fails_fast():
    loop = asyncio.new_event_loop()
    try:
        sched = loop.run_until_complete(_aconstruct())
        with pytest.raises(RuntimeError):
            sched.submit(_req())  # no running loop
    finally:
        loop.close()


def test_submit_from_a_different_loop_fails_fast():
    # The feishu case: a ws thread running its own loop. get_running_loop() would
    # return that loop and not raise; only comparing self._loop catches it.
    home = asyncio.new_event_loop()
    other = asyncio.new_event_loop()
    try:
        sched = home.run_until_complete(_aconstruct())

        async def call():
            sched.submit(_req())

        with pytest.raises(RuntimeError):
            other.run_until_complete(call())
    finally:
        home.close()
        other.close()


async def _aconstruct() -> Scheduler:
    return Scheduler(SuccessRunner(), OriginPools(user=1, system=1), _sink)


# --- conversation_id L1-2 + lane registry ---


async def test_conversation_id_explicit_then_default():
    sched = _scheduler(SuccessRunner())
    assert sched._conversation_id(_req(conversation="cron:7")) == "cron:7"
    assert sched._conversation_id(_req(channel="tg", chat_id="42")) == "tg:42"


async def test_different_conversations_run_concurrently():
    # The reason per-conversation lanes exist: distinct conversations are not
    # serialised behind each other (unlike the old global lock). If they were,
    # only one turn would start and reached would never set -> timeout.
    runner = ConcurrentProbe(target=2)
    sched = Scheduler(runner, OriginPools(user=2, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1"))
    sched.submit(_req(channel="tg", chat_id="2"))
    await asyncio.wait_for(runner.reached.wait(), timeout=1.0)  # both ran at once


async def test_same_conversation_reuses_one_lane():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await sched.submit(_req(channel="tg", chat_id="2")).result()
    assert set(sched._lanes) == {"tg:1", "tg:2"}  # reused tg:1, new tg:2


# --- handle.cancel (per-turn) ---


async def test_handle_cancel_running_turn():
    runner = HangingRunner()
    sched = _scheduler(runner)
    handle = sched.submit(_req())
    await runner.started.wait()
    await handle.cancel()
    assert await asyncio.wait_for(handle.result(), timeout=1.0) is None


async def test_handle_cancel_queued_turn_without_running_it():
    runner = HangingRunner()  # first turn hangs, holding the lane
    sched = _scheduler(runner)
    first = sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1"))  # same lane, queued behind first
    await queued.cancel()
    assert await asyncio.wait_for(queued.result(), timeout=1.0) is None  # resolved, never ran
    assert not first._fut.done()  # surgical: cancelling the queued turn leaves the running one alone


async def test_handle_cancel_is_idempotent_after_completion():
    sched = _scheduler(SuccessRunner())
    handle = sched.submit(_req())
    await handle.result()
    await handle.cancel()  # already terminal -> no-op, no raise


# --- lane reaper (idle recycling + recycle/submit atomicity) ---


async def test_reap_then_resubmit_rebuilds_lane_without_losing_the_request():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)  # let the worker observe the empty queue, exit, and stamp idle_since
    lane = sched._lanes["tg:1"]
    assert lane._idle_since is not None
    assert sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL + 1) == 1  # idle past ttl -> reaped
    assert "tg:1" not in sched._lanes
    handle = sched.submit(_req(channel="tg", chat_id="1"))  # resubmit must not vanish
    assert isinstance(await handle.result(), TurnOutcome)
    assert "tg:1" in sched._lanes and sched._lanes["tg:1"] is not lane  # a fresh lane


async def test_sweep_does_not_reap_a_lane_with_an_inflight_turn():
    runner = HangingRunner()
    sched = _scheduler(runner)
    sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()  # running -> lane active, idle_since is None
    assert sched._sweep(now=1e9) == 0  # a far-future now must still not reap an active lane
    assert "tg:1" in sched._lanes  # the running turn's lane was not yanked from under it


async def test_sweep_keeps_a_lane_still_within_its_idle_ttl():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    assert sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL / 2) == 0  # silent < ttl -> kept
    assert "tg:1" in sched._lanes


async def test_cancel_on_a_reaped_lanes_handle_is_a_noop():
    sched = _scheduler(SuccessRunner())
    handle = sched.submit(_req(channel="tg", chat_id="1"))
    await handle.result()  # turn done; its future resolved
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL + 1)  # reap the lane
    assert "tg:1" not in sched._lanes
    await handle.cancel()  # the handle still references the reaped lane -> no-op, no raise
    assert handle._fut.done()


async def test_resubmit_clears_the_idle_clock_so_a_reused_lane_is_not_reaped():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    assert lane._idle_since is not None  # went idle, clock running
    handle = sched.submit(_req(channel="tg", chat_id="1"))  # reuse before ttl
    assert lane._idle_since is None  # submit reset the clock synchronously
    assert sched._sweep(now=1e9) == 0  # a lane mid-reuse is never reaped
    assert "tg:1" in sched._lanes
    await handle.result()  # drain so the test leaves no pending work


# --- busy policy: permission gate + INTERRUPT ---


class RecordingHangRunner:
    """Records the order turns run in; any turn whose text starts 'hang' blocks
    until cancelled (so an interrupter can preempt it)."""

    def __init__(self):
        self.ran: list[str] = []
        self.hanging_started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.ran.append(req.text)
        if req.text.startswith("hang"):
            self.hanging_started.set()
            await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def _capture_logs() -> tuple[list[str], int]:
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="INFO", format="{message}")
    return lines, sink_id


async def test_non_user_interrupt_is_demoted_to_append_with_a_log():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    try:
        assert sched._effective_busy(_req(origin=Origin.CRON, busy=BusyPolicy.INTERRUPT)) is BusyPolicy.APPEND
        assert sched._effective_busy(_req(origin=Origin.SENTINEL, busy=BusyPolicy.INJECT)) is BusyPolicy.APPEND
    finally:
        logger.remove(sink_id)
    demote = next(line for line in lines if "demoted" in line)
    assert "cron" in demote and "interrupt" in demote and "append" in demote  # origin + requested + applied


async def test_user_busy_policy_is_not_demoted():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    try:
        assert sched._effective_busy(_req(origin=Origin.USER, busy=BusyPolicy.INTERRUPT)) is BusyPolicy.INTERRUPT
        assert sched._effective_busy(_req(origin=Origin.USER, busy=BusyPolicy.INJECT)) is BusyPolicy.INJECT
        assert sched._effective_busy(_req(origin=Origin.CRON, busy=BusyPolicy.APPEND)) is BusyPolicy.APPEND
    finally:
        logger.remove(sink_id)
    assert not any("demoted" in line for line in lines)  # no demotion -> no log


async def test_non_user_interrupt_does_not_preempt_the_running_turn():
    # A proactive origin must not be able to kill a user's turn.
    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    user = sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    sched.submit(_req(channel="tg", chat_id="1", origin=Origin.CRON, busy=BusyPolicy.INTERRUPT, text="cron"))
    await asyncio.sleep(0)
    assert not user._fut.done()  # the user's running turn was not preempted


async def test_user_interrupt_preempts_then_runs_before_appended_backlog():
    runner = RecordingHangRunner()
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    hang = sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    appended = sched.submit(_req(channel="tg", chat_id="1", text="append"))  # queued behind
    interrupter = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="interrupt"))
    assert await asyncio.wait_for(hang.result(), timeout=1.0) is None  # preempted
    await asyncio.wait_for(interrupter.result(), timeout=1.0)
    await asyncio.wait_for(appended.result(), timeout=1.0)
    assert runner.ran == ["hang", "interrupt", "append"]  # interrupter jumped ahead of the backlog
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)  # preemption disclosed


async def test_two_interrupts_run_latest_first():
    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    i1 = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="i1"))
    i2 = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="i2"))
    await asyncio.wait_for(i1.result(), timeout=1.0)
    await asyncio.wait_for(i2.result(), timeout=1.0)
    assert runner.ran == ["hang", "i2", "i1"]  # latest interrupt runs first


async def test_interrupt_on_an_idle_lane_runs_like_append():
    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    handle = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="x"))
    assert isinstance(await asyncio.wait_for(handle.result(), timeout=1.0), TurnOutcome)
    assert runner.ran == ["x"]  # nothing to preempt -> just runs


# --- busy policy: INJECT (mailbox / merge-and-chain / fallback / fate split) ---


class MergingRunner:
    """Drains the inject mailbox once at a simulated tool-loop gap and records the
    merged content; the host turn then completes with its outcome."""

    def __init__(self, gate, outcome):
        self.started = asyncio.Event()
        self.merged: list[str] = []
        self._gate = gate
        self._outcome = outcome

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.started.set()
        await self._gate.wait()
        self.merged.extend(r.text for r in drain())
        return self._outcome


class NonDrainingRunner:
    """The minimal-legal runner: never drains, so any inject must fall back."""

    def __init__(self, gate):
        self.started = asyncio.Event()
        self.ran: list[str] = []
        self._gate = gate

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.ran.append(req.text)
        self.started.set()
        await self._gate.wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


async def test_inject_merges_into_host_and_chains_its_outcome():
    gate = asyncio.Event()
    outcome = TurnOutcome(usage=Usage(3, 4, 7), explicit_reply=True)
    runner = MergingRunner(gate, outcome)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    gate.set()  # host resumes, drains+merges the inject, completes
    assert await asyncio.wait_for(host.result(), timeout=1.0) == outcome
    assert await asyncio.wait_for(inject.result(), timeout=1.0) == outcome  # result chained to host
    assert runner.merged == ["inject"]


async def test_cancel_on_a_merged_inject_is_noop_and_host_survives():
    drain_gate, finish_gate = asyncio.Event(), asyncio.Event()
    outcome = TurnOutcome(usage=Usage(1, 1, 2), explicit_reply=True)

    class DrainThenWaitRunner:
        def __init__(self):
            self.started = asyncio.Event()
            self.drained = asyncio.Event()

        async def run(self, req, emit, drain) -> TurnOutcome:
            self.started.set()
            await drain_gate.wait()
            drain()  # merge -> the inject is now chained to this host
            self.drained.set()
            await finish_gate.wait()  # host keeps running after the merge
            return outcome

    runner = DrainThenWaitRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    drain_gate.set()
    await runner.drained.wait()
    await inject.cancel()  # already merged (chained), host still running -> no-op, must not kill host
    finish_gate.set()
    assert await asyncio.wait_for(host.result(), timeout=1.0) == outcome  # host survived
    assert await asyncio.wait_for(inject.result(), timeout=1.0) == outcome  # chained outcome


async def test_undrained_inject_falls_back_to_append_with_a_log():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    lines, sink_id = _capture_logs()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    gate.set()  # host completes WITHOUT draining -> the inject falls back to APPEND
    try:
        assert isinstance(await asyncio.wait_for(inject.result(), timeout=1.0), TurnOutcome)
    finally:
        logger.remove(sink_id)
    assert runner.ran == ["host", "inject"]  # inject ran as a fresh turn after the host
    assert any("fell back" in line for line in lines)  # USER inject fallback is logged


async def test_stop_clears_the_inject_mailbox_as_cancelled():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    stopped = sched._lanes["tg:1"].cancel()
    assert stopped == 2  # running host + 1 undrained inject
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None  # cancelled, not fallen back
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    assert runner.ran == ["host"]  # the inject never ran


async def test_cancel_on_a_mailboxed_inject_removes_it_before_merge():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    await inject.cancel()  # still in the mailbox -> self-cancel (removed, resolved cancelled)
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None
    assert len(sched._lanes["tg:1"]._inject_mailbox) == 0
    gate.set()
    await asyncio.wait_for(host.result(), timeout=1.0)
    assert runner.ran == ["host"]  # the cancelled inject did not fall back / run


async def test_interrupt_runs_before_a_hosts_fallen_back_inject():
    # Ordering: the interrupter jumps to the front; the preempted host's
    # undrained inject falls back to the tail, so it runs after the interrupter.
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    interrupter = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="interrupt"))
    gate.set()  # lets the interrupter and the fallen-back inject finish
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None  # preempted
    await asyncio.wait_for(interrupter.result(), timeout=1.0)
    await asyncio.wait_for(inject.result(), timeout=1.0)
    assert runner.ran == ["host", "interrupt", "inject"]  # interrupter first, fallback inject last


async def test_stop_after_drain_resolves_a_chained_inject_exactly_once():
    # The double-resolve hazard: a drained inject is chained to the host. /stop
    # must not also resolve it from the mailbox (drain removed it) — it follows
    # the host and resolves exactly once, never raising InvalidStateError.
    drain_gate = asyncio.Event()

    class R:
        def __init__(self):
            self.started = asyncio.Event()
            self.drained = asyncio.Event()

        async def run(self, req, emit, drain) -> TurnOutcome:
            self.started.set()
            await drain_gate.wait()
            drain()  # the inject is now chained to this host
            self.drained.set()
            await asyncio.Event().wait()  # host hangs after the merge
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    runner = R()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    drain_gate.set()
    await runner.drained.wait()  # inject drained -> chained, host still running
    stopped = sched._lanes["tg:1"].cancel()
    assert stopped == 1  # only the running host; the chained inject is not re-counted from the mailbox
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None  # followed the host, resolved once


# --- shutdown (seal / drain / grace / cascade) + reaper driver ---


async def test_shutdown_seals_the_scheduler_then_submit_raises():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    await sched.shutdown(grace=1.0)
    try:
        with pytest.raises(SchedulerDrainingError):
            sched.submit(_req())
    finally:
        logger.remove(sink_id)
    assert any("draining" in line for line in lines)  # rejection logged


async def test_shutdown_drains_unstarted_work_and_lets_the_running_turn_finish():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = _scheduler(runner)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1", text="queued"))  # APPEND, queued
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inj"))  # mailbox
    gate.set()  # host can finish within grace
    await sched.shutdown(grace=1.0)
    assert isinstance(await host.result(), TurnOutcome)  # finished in grace -> normal terminal
    assert await queued.result() is None  # unstarted -> drained cancelled
    assert await inject.result() is None  # mailbox -> drained cancelled
    assert runner.ran == ["host"]  # the queued turn never ran


async def test_shutdown_cancels_a_running_turn_that_outlasts_grace():
    runner = HangingRunner()
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()
    await sched.shutdown(grace=0.05)  # host hangs past grace -> cascade-cancel
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None  # resolved, not hung
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)


async def test_reaper_lazy_starts_on_first_submit_and_shutdown_cancels_it():
    sched = _scheduler(SuccessRunner())
    assert sched._reaper is None  # not started before any submit
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    reaper = sched._reaper
    assert reaper is not None and not reaper.done()  # lazy-started alongside the first lane
    await sched.shutdown(grace=1.0)
    assert reaper.done()  # shutdown cancelled it


async def test_reap_loop_self_terminates_when_no_lanes():
    sched = _scheduler(SuccessRunner())
    await asyncio.wait_for(sched._reap_loop(), timeout=1.0)  # no lanes -> exits at once, never sleeps


async def test_running_reaper_actually_sweeps_and_reaps_an_idle_lane(monkeypatch):
    # The reaper-driver requirement: the running reaper must actually call _sweep
    # and reclaim a silent lane (not merely exist). Shrink the interval and the
    # idle ttl so a real sweep reaps within the test.
    monkeypatch.setattr("raven.spine.scheduler._SWEEP_INTERVAL", 0.01)
    monkeypatch.setattr("raven.spine.scheduler._DEFAULT_IDLE_TTL", 0.0)
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()  # runs, then goes idle
    for _ in range(100):
        if "tg:1" not in sched._lanes:
            break
        await asyncio.sleep(0.01)
    assert "tg:1" not in sched._lanes  # the running reaper swept and reaped it (not manual _sweep)


async def test_shutdown_on_an_idle_scheduler_is_a_clean_noop():
    sched = _scheduler(SuccessRunner())
    await sched.shutdown(grace=1.0)  # no lanes / no reaper / nothing running -> clean no-op
    with pytest.raises(SchedulerDrainingError):
        sched.submit(_req())  # still sealed


# --- lane queue depth warning ---


async def test_submit_backlog_warns_when_pending_reaches_the_threshold(monkeypatch):
    monkeypatch.setattr("raven.spine.scheduler._DEPTH_WARN_THRESHOLD", 3)
    runner = HangingRunner()  # host hangs, holding the lane so submits queue
    sched = _scheduler(runner)
    lines, sink_id = _capture_logs()
    try:
        sched.submit(_req(channel="tg", chat_id="1", text="host"))  # runs, hangs (pending depth 0)
        await runner.started.wait()
        sched.submit(_req(channel="tg", chat_id="1", text="q1"))  # depth 1
        sched.submit(_req(channel="tg", chat_id="1", text="q2"))  # depth 2
        assert not any("depth" in line for line in lines)  # below threshold -> no warning
        sched.submit(_req(channel="tg", chat_id="1", text="q3"))  # depth 3 == threshold -> warn
    finally:
        logger.remove(sink_id)
    warn = next(line for line in lines if "depth" in line)
    assert "tg:1" in warn and "3" in warn  # conversation id + depth
    await sched.shutdown(grace=0.01)  # clean teardown


async def test_inject_fallback_re_enqueue_also_triggers_the_depth_warning(monkeypatch):
    # The hole the reviewer caught: fallback re-append bypasses submit. Because it
    # routes through _enqueue too (one +1 per inject), the == depth check still
    # fires when the fallback grows the queue past the threshold.
    monkeypatch.setattr("raven.spine.scheduler._DEPTH_WARN_THRESHOLD", 3)
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)  # never drains -> injects fall back at turn end
    sched = _scheduler(runner)
    lines, sink_id = _capture_logs()
    try:
        sched.submit(_req(channel="tg", chat_id="1", text="host"))  # runs, awaits gate
        await runner.started.wait()
        injects = [
            sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text=f"inj{i}")) for i in range(3)
        ]
        gate.set()  # host completes -> worker re-enqueues the 3 injects via _enqueue (depth -> 3)
        for handle in injects:
            await asyncio.wait_for(handle.result(), timeout=1.0)  # fallback turns ran
    finally:
        logger.remove(sink_id)
    assert any("depth" in line and "3" in line for line in lines)  # fallback drove the warning
    await sched.shutdown(grace=0.01)


def _turn_entry_strays(package_root):
    """Every call of ``run_turn`` / ``_run_turn`` outside the sanctioned
    entries, as ``file:line`` strays. The roster is the point: a turn enters
    through ``Scheduler.submit``, which drives the loop through one of the
    three spine adapters -- any other caller starts a turn the scheduler
    cannot order, cancel, or swap around, and the retry test below already
    names that bypass as the hole nobody watched.
    """
    import ast

    allowed_run_turn = {
        "raven/agent/spine_runner.py",
        "raven/rpc/spine.py",
        "raven/gateway/spine.py",
        # The fourth adapter: an inbound A2A task is one turn, run for a peer
        # agent with no watcher and no conversation of its own, the way
        # rpc/spine.py's CRON branch runs one for a cron job. It joins the
        # roster rather than hiding behind a rename, because the cost is real
        # and belongs written down: a turn started here is one the scheduler
        # cannot order, cancel or swap, so a peer's concurrency is bounded by
        # nothing on this path.
        "raven/a2a/runtime.py",
    }
    allowed_private = {
        "raven/spine/scheduler.py",
        "raven/agent/loop/main.py",
        "raven/agent/loop/turn_path.py",
    }
    strays = []
    for p in sorted(package_root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(package_root.parent).as_posix()
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "run_turn" and rel not in allowed_run_turn:
                strays.append(f"{rel}:{node.lineno} calls run_turn outside the spine adapters")
            if node.func.attr == "_run_turn" and rel not in allowed_private:
                strays.append(f"{rel}:{node.lineno} calls _run_turn outside its own class")
    return strays


def test_every_turn_enters_through_submit():
    """The static half of the ordering promise. tests/ excluded on purpose --
    a test drives turns directly by design; production code never may."""
    from pathlib import Path

    strays = _turn_entry_strays(Path(__file__).resolve().parents[1] / "raven")
    assert strays == [], (
        "a turn must enter through Scheduler.submit; a new runner adapter joins "
        "the roster in _turn_entry_strays in the same change:\n" + "\n".join(strays)
    )


def test_the_turn_entry_guard_bites(tmp_path):
    """Mutation audit, built in: a stray caller in a synthetic tree turns the
    checker red, so a green real-tree run means looked-and-found-nothing."""
    pkg = tmp_path / "raven"
    pkg.mkdir()
    (pkg / "sneaky.py").write_text(
        "async def go(loop, req):\n    await loop.run_turn(req)\n    await loop._run_turn(req)\n",
        encoding="utf-8",
    )
    strays = _turn_entry_strays(pkg)
    assert len(strays) == 2, strays


# --- settlement under cancellation races -------------------------------------
#
# A consumer that awaited TurnHandle.result() inside a task and was itself
# cancelled -- a CLI interrupt, an aborted gateway request, a caller that gave up
# -- cancels the future it was awaiting. The lane's own resolver then meets a dead
# future, and every path below used to call set_result on it. These pin the
# invariants that cost: the entry leaves the queue, the turn's end is still named
# exactly once, the lane worker survives to run the next turn, and nothing raises.


_SETTLE_TIMEOUT = 1.0


def _recorder() -> tuple[list, object]:
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    return events, sink


def _terminal(events: list, turn_id: str) -> list:
    """A turn's own terminal events. Exactly one is the contract: a consumer
    releases the ending turn's slots on the first, and a second is exactly what
    per-turn idempotent release exists to absorb."""
    return [e for e in events if isinstance(e, (TurnEnded, TurnFailed)) and e.turn_id == turn_id]


def _names(events: list) -> list[str]:
    return [type(e).__name__ for e in events]


async def _settled(handle) -> object:
    """Wait for a turn's future to be terminal without letting a cancelled
    waiter's CancelledError escape into the test. Bounded, so a regression that
    never settles fails here instead of hanging the suite."""
    fut = handle._fut
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SETTLE_TIMEOUT
    while not fut.done():
        if loop.time() > deadline:
            pytest.fail("turn future never settled")
        await asyncio.sleep(0.001)
    return None if fut.cancelled() else fut.result()


async def _wait_task_done(task) -> None:
    """Bounded wait for a task to finish. Polling rather than awaiting it, so a
    regression that strands the task reports 'worker never terminated' instead of
    having the wait itself cancel the evidence."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SETTLE_TIMEOUT
    while not task.done():
        if loop.time() > deadline:
            pytest.fail("lane worker never terminated")
        await asyncio.sleep(0.001)


def _worker_error(lane) -> BaseException | None:
    """What the lane worker died of, if it is already finished. The sentinel for
    every test in this section: an InvalidStateError escaping a settlement path
    lands here, and on a still-running worker it is an unretrieved task exception
    that pytest would report without saying where."""
    task = lane._worker
    if task is None or not task.done() or task.cancelled():
        return None
    return task.exception()


def _assert_released(lane) -> None:
    assert lane._run_task is None and lane._running_fut is None  # the in-flight slot is free


async def test_queued_future_cancelled_before_drain_leaves_the_queue_clean():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)  # completes once the gate opens
    events, sink = _recorder()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host", turn_id="host"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1", text="queued", turn_id="queued"))
    lane = sched._lanes["tg:1"]

    queued._fut.cancel()  # its submitter's task went away while the turn sat queued
    assert queued._fut.cancelled()
    assert len(lane._pending) == 1  # nothing has released the entry yet

    assert lane.drain_pending() == 1  # /stop's queue drain
    assert not lane._pending  # the entry is gone, not left to run behind the host
    assert queued._fut.cancelled()  # settled once: cancelled, never re-resolved

    gate.set()
    assert isinstance(await _settled(host), TurnOutcome)  # the running turn is untouched
    assert _names(_terminal(events, "host")) == ["TurnEnded"]
    assert _terminal(events, "queued") == []  # a drained turn never ran; drain is silent by design
    later = sched.submit(_req(channel="tg", chat_id="1", text="later", turn_id="later"))
    assert isinstance(await _settled(later), TurnOutcome)  # the lane is still usable
    assert runner.ran == ["host", "later"]  # the cancelled queued turn never ran
    _assert_released(lane)
    assert _worker_error(lane) is None


async def test_running_future_cancelled_while_the_worker_completes():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    events, sink = _recorder()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    first = sched.submit(_req(channel="tg", chat_id="1", text="first", turn_id="first"))
    await runner.started.wait()
    lane = sched._lanes["tg:1"]
    second = sched.submit(_req(channel="tg", chat_id="1", text="second", turn_id="second"))  # queued behind

    first._fut.cancel()  # the submitter's task went away mid-turn
    gate.set()  # the worker completes the turn and resolves a dead future
    # Without the guard the set_result in the worker's finally raises
    # InvalidStateError out of that finally, killing the worker mid-drain: the
    # mailbox fallback below it never runs and second never starts, so this
    # wait_for is the bounded failure.
    assert isinstance(await _settled(second), TurnOutcome)
    assert first._fut.cancelled()  # settled once: the dead future was not revived
    assert _names(_terminal(events, "first")) == ["TurnEnded"]  # the end is still named
    assert _names(_terminal(events, "second")) == ["TurnEnded"]
    assert runner.ran == ["first", "second"]
    _assert_released(lane)
    assert _worker_error(lane) is None


async def test_handle_cancel_survives_its_own_future_being_cancelled_mid_call():
    # The race inside cancel_turn: its queued branch awaits the sink before it
    # resolves, and the caller's future can be cancelled across that await.
    parked, release = asyncio.Event(), asyncio.Event()
    events: list = []

    async def sink(event) -> None:
        events.append(event)
        if isinstance(event, TurnFailed):
            parked.set()
            await release.wait()  # park inside cancel_turn's terminal emit

    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host", turn_id="host"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1", text="queued", turn_id="queued"))
    lane = sched._lanes["tg:1"]

    cancelling = asyncio.create_task(queued.cancel())
    await asyncio.wait_for(parked.wait(), timeout=_SETTLE_TIMEOUT)  # parked inside the sink await
    queued._fut.cancel()  # the submitter's task goes away across that await
    release.set()
    await asyncio.wait_for(cancelling, timeout=_SETTLE_TIMEOUT)  # returned instead of raising
    assert queued._fut.cancelled()  # settled once: cancelled, not re-resolved
    assert not lane._pending  # the entry was removed before the emit, as it always was
    assert _names(_terminal(events, "queued")) == ["TurnFailed"]  # named exactly once
    assert not host._fut.done()  # surgical: the running turn is left alone

    gate.set()
    assert isinstance(await _settled(host), TurnOutcome)
    assert _names(_terminal(events, "host")) == ["TurnEnded"]
    later = sched.submit(_req(channel="tg", chat_id="1", text="later", turn_id="later"))
    assert isinstance(await _settled(later), TurnOutcome)  # the lane is still usable
    assert runner.ran == ["host", "later"]
    _assert_released(lane)
    assert _worker_error(lane) is None


async def test_shutdown_after_an_interrupted_waiter_still_files_the_turns_end():
    # The CLI interrupt, end to end: the submitter's task is cancelled while
    # parked on the handle, so the future goes with it, and shutdown then has to
    # cascade-cancel a turn whose future is already dead. The turn's end is still
    # owed to its consumer -- the turn had not reported when the waiter left.
    runner = HangingRunner()
    events, sink = _recorder()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host", turn_id="host"))
    await runner.started.wait()
    lane = sched._lanes["tg:1"]
    worker = lane._worker

    waiter = asyncio.create_task(host.result())  # the submitter is parked on the handle
    await asyncio.sleep(0.001)  # let it reach the await
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter
    assert host._fut.cancelled()  # the interrupt reached the future the lane resolves

    await asyncio.wait_for(sched.shutdown(grace=0.01), timeout=2 * _SETTLE_TIMEOUT)  # outlasts grace
    await _wait_task_done(worker)
    # The regression the previous candidate shipped as green: without the
    # cascade-cancel actually reaching the payload, this list is empty and the
    # turn ends with nothing named -- a consumer holding its slots forever.
    assert _names(_terminal(events, "host")) == ["TurnFailed"]
    assert host._fut.cancelled()  # settled once, not revived by the cascade
    assert worker.cancelled() is False
    assert worker.exception() is None  # no InvalidStateError out of the worker's finally
    with pytest.raises(SchedulerDrainingError):
        sched.submit(_req(channel="tg", chat_id="2"))  # shutdown seals the lane for good


async def test_completion_and_cancel_racing_resolve_a_turn_exactly_once():
    # Both winners, twelve rounds: on the even rounds the caller cancels in the
    # window the guard exists for -- the turn has filed its TurnEnded and the
    # lane has not resolved the handle yet -- and on the odd rounds the
    # completion wins and the later cancel is a no-op.
    rounds = 12
    for round_no in range(rounds):
        events: list = []
        holder: dict = {}

        async def sink(event, _events=events, _holder=holder, _cancel_wins=round_no % 2 == 0) -> None:
            _events.append(event)
            if _cancel_wins and isinstance(event, TurnEnded):
                _holder["handle"]._fut.cancel()

        sched = Scheduler(SuccessRunner(), OriginPools(user=1, system=1), sink)
        handle = sched.submit(_req(channel="tg", chat_id="1", text="t", turn_id=f"t{round_no}"))
        holder["handle"] = handle
        lane = sched._lanes["tg:1"]
        worker = lane._worker

        await _settled(handle)
        if round_no % 2 == 0:
            assert handle._fut.cancelled()  # the caller won the race
        else:
            assert handle._fut.cancel() is False  # completion won: the cancel is a no-op
        await _wait_task_done(worker)
        assert worker.cancelled() is False
        assert worker.exception() is None  # neither order raised out of the settlement
        assert _names(_terminal(events, f"t{round_no}")) == ["TurnEnded"]  # named exactly once
        _assert_released(lane)
        assert _worker_error(lane) is None


async def test_cancelled_mailbox_inject_drains_clean_and_leaves_the_host_running():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)  # never drains, so the inject stays mailboxed
    events, sink = _recorder()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host", turn_id="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject", turn_id="inject"))
    lane = sched._lanes["tg:1"]
    assert len(lane._inject_mailbox) == 1

    inject._fut.cancel()  # its submitter's task went away while it sat in the mailbox
    assert inject._fut.cancelled()

    assert lane.drain_pending() == 1  # /stop's drain, with the entry already dead
    assert not lane._inject_mailbox  # the entry is gone
    assert inject._fut.cancelled()  # settled once: cancelled, never revived
    assert not host._fut.done()  # cancelling an inject does not touch its host

    gate.set()
    assert isinstance(await _settled(host), TurnOutcome)
    assert _names(_terminal(events, "host")) == ["TurnEnded"]
    # An inject binds no slots of its own, so its own cancellation is silent --
    # a terminal event here would announce an end on the host's lane while the
    # host is still going.
    assert _terminal(events, "inject") == []
    later = sched.submit(_req(channel="tg", chat_id="1", text="later", turn_id="later"))
    assert isinstance(await _settled(later), TurnOutcome)  # the lane is still usable
    assert runner.ran == ["host", "later"]  # the cancelled inject never fell back, never ran
    _assert_released(lane)
    assert _worker_error(lane) is None


async def test_merged_inject_whose_waiter_was_cancelled_still_lets_the_host_end():
    # The other half of the mailbox: the inject was drained and chained to the
    # host, so the host's own finally is its resolver -- and the host's TurnEnded
    # sits on the far side of that settlement.
    drained, finish = asyncio.Event(), asyncio.Event()
    merge = asyncio.Event()
    outcome = TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)

    class DrainThenWaitRunner:
        def __init__(self):
            self.started = asyncio.Event()
            self.drained = asyncio.Event()

        async def run(self, req, emit, drain) -> TurnOutcome:
            self.started.set()
            await merge.wait()  # the submitter files the inject before the gap it merges at
            drain()  # merge now: the inject is chained to this host
            self.drained.set()
            await finish.wait()
            return outcome

    runner = DrainThenWaitRunner()
    events, sink = _recorder()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host", turn_id="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject", turn_id="inject"))
    merge.set()
    await asyncio.wait_for(runner.drained.wait(), timeout=_SETTLE_TIMEOUT)
    lane = sched._lanes["tg:1"]
    assert not lane._inject_mailbox  # merged: the mailbox no longer owns it

    inject._fut.cancel()  # the inject's submitter gave up after the merge
    finish.set()
    assert await _settled(host) == outcome  # the host ended normally, not by raising
    assert _names(_terminal(events, "host")) == ["TurnEnded"]  # its report survived the settlement
    assert inject._fut.cancelled()  # settled once, not re-resolved with the host's outcome
    later = sched.submit(_req(channel="tg", chat_id="1", text="later", turn_id="later"))
    assert isinstance(await _settled(later), TurnOutcome)  # the lane is still usable
    _assert_released(lane)
    assert _worker_error(lane) is None


async def test_settle_guard_passes_through_an_unrelated_invalid_state():
    # The guard is not a blanket suppress: an InvalidStateError on a future that
    # is NOT done is a real defect -- a resolver holding a future it does not own
    # -- and must still reach the caller rather than be swallowed as settled.
    class Wrong:
        """A stand-in whose done() lies: done, but not actually settled."""

        def __init__(self):
            self.calls = 0

        def done(self) -> bool:
            return True

        def set_result(self, value) -> None:
            self.calls += 1
            raise asyncio.InvalidStateError

    already_terminal = Wrong()
    _settle(already_terminal, None)  # short-circuits on done(): never reaches set_result
    assert already_terminal.calls == 0

    class Unowned:
        """A future that is not done and still refuses to be settled: the second
        failure mode, and the one the guard must not swallow."""

        def __init__(self):
            self.calls = 0

        def done(self) -> bool:
            return False

        def set_result(self, value) -> None:
            self.calls += 1
            raise asyncio.InvalidStateError

    unowned = Unowned()
    with pytest.raises(asyncio.InvalidStateError):
        _settle(unowned, None)
    assert unowned.calls == 1  # raised out, not swallowed as settled
