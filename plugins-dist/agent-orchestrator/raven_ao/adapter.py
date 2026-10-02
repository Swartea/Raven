"""The Raven -> AO CodeRun adapter: submit, status, result, cancel.

Design rules this module implements, and which the tests pin down:

*Raven holds no run state.* The only thing persisted is the mapping from a
Raven step to the AO ``request_id`` that controls it. Every question about how a
run is going is asked of AO on demand.

*Control is by ``request_id``.* AO's CLI takes a positional ``request_id`` on
``code-run-status``, ``code-run-result`` and ``code-run-cancel`` alike, and
``code-run-submit`` keys its durable record by it. The adapter therefore never
needs a ``run_id`` to drive a run, and never requires one to exist: a submission
whose ``run_id`` AO has not published yet is still fully controllable.

*Submit is idempotent.* The mapping is reserved from the parent identifiers
*before* the AO call. A repeat submit, a reconnect after a restart, or two Raven
processes racing on the same step all find the existing mapping and reuse its
``request_id`` -- none of them launches a second CodeRun. A submit whose request
contents changed is reported, not silently relaunched.

*Acceptance comes from AO's structured verdict only.* ``result`` will surface an
accepted commit or patch only when AO's own ``change_acceptance`` is
``accepted`` and the same formal response carries a concrete artifact with
``verification_complete`` true. A zero exit code, a terminal execution state, or
Raven's own bookkeeping never promotes a result.

*No auto-retry.* A terminal or blocked run is reported as-is.

*AO's event log is read-only truth.* ``events`` consumes AO's own
``code-run-events REQUEST_ID --after-seq N`` contract. Raven persists only a
cursor -- the highest sequence it has consumed -- and a projection of the
validated page. It never rewrites the log, never reconstructs a missing event,
and never infers execution state, acceptance, or terminality from an event: the
log is what AO published, and the verdict still comes from a formal result.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Final

from raven_ao.client import AoCliClient, CliInvocationError, CliResult
from raven_ao.config import AdapterConfig
from raven_ao.contract import (
    ACCEPTANCE_ACCEPTED,
    MalformedResponseError,
    cli_error_detail,
    lifecycle,
    parse_response,
    validate_result_shape,
    validate_status_shape,
)
from raven_ao.mapping import (
    CURSOR_BEFORE_FIRST_EVENT,
    STATE_CANCEL_REQUESTED,
    STATE_RESERVED,
    STATE_STALE_MAPPING,
    STATE_SUBMITTED,
    CapacityGuard,
    MappingStore,
    RunMapping,
    derive_request_id,
    fingerprint_request,
)

#: Identifiers the adapter owns and injects into every submitted request. AO
#: carries all three on both ``CodeRunRequest`` and ``CodeRunResult``, so a
#: package submitted without them cannot be traced back to the Raven step that
#: asked for it.
_INJECTED_FIELDS = ("request_id", "parent_goal_id", "parent_step_id")

#: An AO event sequence is a non-negative, strictly increasing integer: AO
#: numbers its durable event log from ``0``. It is read as a decimal string or a
#: number, never as a float, so a fractional or exponent-formatted sequence is
#: rejected instead of silently truncated.
_SEQ_RE: Final = re.compile(r"^[0-9]+$")

#: AO's ``code-run-events`` cursor form: an explicit signed decimal, so the
#: ``-1`` that precedes event ``0`` passes through unchanged.
_CURSOR_RE: Final = re.compile(r"^-?[0-9]+$")


#: How long a submit waits between attempts to take a request's
#: control lock. Short enough to add no real latency, long enough not to spin.
CONTROL_LOCK_POLL_SECONDS: Final = 0.005


def _parse_event_timestamp(value: Any) -> str:
    """Return AO's timestamp verbatim, or reject it as unparseable.

    AO emits ISO-8601 and may use the trailing ``Z`` form, which
    :meth:`datetime.datetime.fromisoformat` does not accept on its own before
    Python 3.11, so the suffix is normalized for the parse only -- the value
    Raven reports back is always AO's original string, never a reformatted one.

    Shape is not enough: a real calendar date and a real clock time are
    required, so ``2026-13-45T99:00:00Z`` is rejected instead of being carried
    forward as an unorderable projection.
    """
    if not isinstance(value, str) or not value.strip():
        raise MalformedResponseError(
            "event field 'timestamp' was not a non-empty string"
        )
    candidate = value.strip()
    parseable = candidate[:-1] + "+00:00" if candidate.endswith(("Z", "z")) else candidate
    try:
        datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise MalformedResponseError(
            f"event timestamp {value!r} is not a valid ISO-8601 date and time"
        ) from exc
    return value


class AdapterError(RuntimeError):
    """A user-facing adapter failure (bad arguments, no such run, bad config)."""


def _stamp() -> str:
    return datetime.now(UTC).isoformat()


def effective_request(
    parent_goal_id: str,
    parent_step_id: str,
    request: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Return the request actually submitted, with Raven's ids injected.

    AO's ``CodeRunRequest`` carries ``request_id``, ``parent_goal_id`` and
    ``parent_step_id``, so the adapter is the one that has to supply them: the
    caller knows which Raven step this is, and ``request_id`` must be the stable
    id derived from exactly that step or a reconnect cannot find the run.

    A caller-supplied value that *disagrees* is refused rather than overwritten.
    Silently replacing a mismatched ``request_id`` would submit a package AO
    files under one id while Raven tracks another, and the two would never meet
    again.
    """
    effective = dict(request)
    expected = {
        "request_id": request_id,
        "parent_goal_id": parent_goal_id,
        "parent_step_id": parent_step_id,
    }
    for field in _INJECTED_FIELDS:
        supplied = effective.get(field)
        if supplied is not None and supplied != expected[field]:
            raise AdapterError(
                f"request {field} {supplied!r} does not match this Raven step "
                f"({expected[field]!r}); refusing to submit a mismatched package"
            )
        effective[field] = expected[field]
    return effective


def _formal(cli_result: CliResult, command: str) -> dict[str, Any]:
    """Turn a CLI outcome into a formal payload, or raise.

    A non-zero exit or unparseable stdout is a CLI failure, reported as such.
    It is never converted into a verdict about the run.
    """
    if not cli_result.ok:
        raise CliInvocationError(
            f"AO CLI '{command}' exited {cli_result.exit_code}: "
            f"{cli_error_detail(cli_result.stdout, cli_result.stderr)[:400]}"
        )
    return parse_response(cli_result.stdout)


class CodeRunAdapter:
    """Durable Raven -> AO CodeRun adapter."""

    def __init__(self, config: AdapterConfig, client: AoCliClient | None = None) -> None:
        self._config = config
        self._client = client or AoCliClient(
            config.executable,
            runs_dir=config.runs_dir,
            working_dir=config.working_dir,
            timeout=config.timeout,
        )
        self._store = MappingStore(config.mapping_dir)
        self._capacity = CapacityGuard(config.mapping_dir)

    @property
    def store(self) -> MappingStore:
        return self._store

    @property
    def capacity(self) -> CapacityGuard:
        return self._capacity

    @property
    def max_concurrent_code_runs(self) -> int:
        return self._config.max_concurrent_code_runs

    @property
    def client(self) -> AoCliClient:
        return self._client

    # -- capacity --------------------------------------------------------
    def _release_capacity(self, request_id: str) -> bool:
        """Release a run's outer-AO slot once AO says it can no longer need it.

        Called only from an authoritative AO answer -- a terminal status or
        result, or AO explicitly confirming the request is missing. A quiet run,
        a failed CLI invocation, and an event page all leave the slot held: only
        AO knows the run is finished.
        """
        return self._capacity.release(request_id)

    async def _release_after_confirmed_missing(self, mapping: RunMapping) -> bool:
        """Release a slot only once AO confirms the request is not there.

        Used after AO refuses a submission: the refusal is not by itself proof
        that no run was created, so the durable status read decides. If the
        status read fails or still shows the request, the slot stays held.
        """
        try:
            status_result = await self._client.status(mapping.request_id)
        except CliInvocationError:
            return False
        if not _is_missing_request(status_result):
            return False
        return self._release_capacity(mapping.request_id)

    # -- submit ---------------------------------------------------------
    async def submit(
        self, parent_goal_id: str, parent_step_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit a CodeRun for a Raven coding step, at most once.

        The adapter injects the stable ``request_id`` and both parent
        identifiers into the supplied request, then sends it to AO as a file.
        The fingerprint is taken over the request that was *actually* submitted,
        so a later submit that differs in any injected or caller-supplied field
        is detectable.
        """
        if not parent_goal_id or not parent_step_id:
            raise AdapterError("parent_goal_id and parent_step_id are required")
        if not isinstance(request, dict):
            raise AdapterError("CodeRunRequest must be a JSON object")

        request_id = derive_request_id(parent_goal_id, parent_step_id)
        # One Raven process at a time per identity, across processes. Without
        # this, a second process can observe the mapping as absent, share the
        # same capacity marker, and reach its recovery status probe before the
        # first process's submit has registered with AO. AO would not know the
        # request yet and would answer CodeRunNotFound, which the probe would
        # then report as a stale mapping and act on by releasing the first
        # call's still-running slot. Serializing the whole reserve/submit/
        # decide sequence means the second process instead observes the first
        # one's finished mapping and reuses it.
        async with self._control_lock(request_id):
            return await self._submit_locked(parent_goal_id, parent_step_id, request, request_id)

    @asynccontextmanager
    async def _control_lock(self, request_id: str) -> AsyncIterator[None]:
        """Hold the per-request control lock, waiting without blocking the loop.

        Acquisition is non-blocking and polled with a yield between attempts, so
        a second coroutine waiting for the same identity stays responsive and can
        still run the process that holds the lock -- which is the whole point of
        the lock. A blocking :func:`flock` would instead stop the event loop for
        every other task until the holder released it.
        """
        fd, acquired = self._store.try_acquire_control(request_id)
        while not acquired:
            await asyncio.sleep(CONTROL_LOCK_POLL_SECONDS)
            fd, acquired = self._store.try_acquire_control(request_id)
        try:
            yield
        finally:
            self._store.release_control(fd)

    async def _submit_locked(
        self,
        parent_goal_id: str,
        parent_step_id: str,
        request: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Run one submit decision sequence under the request's control lock."""
        candidate = RunMapping(
            request_id=request_id,
            parent_goal_id=parent_goal_id,
            parent_step_id=parent_step_id,
            request_fingerprint=fingerprint_request(
                effective_request(parent_goal_id, parent_step_id, request, request_id)
            ),
        )
        # Capacity is claimed *before* the mapping is written, so a step that
        # cannot get a slot leaves no request mapping behind at all. Creating
        # one would make the blocked attempt look like a launched run to a later
        # reader; deferring leaves the step untouched, so a later explicit call
        # simply retries once capacity frees.
        limit = self._config.max_concurrent_code_runs
        already_mapped = self._store.get(request_id)
        if already_mapped is None:
            acquired, active = self._capacity.acquire(candidate.request_id, limit)
            if not acquired:
                return {
                    **_submit_payload(candidate, launched=False),
                    "ok": False,
                    "outcome": "capacity_exhausted",
                    "limit": limit,
                    "active_code_runs": active,
                    "mapping_created": False,
                    "detail": (
                        f"Raven's outer AO concurrency limit is {limit} and that many "
                        "CodeRuns are already active. No run was launched and no "
                        "request mapping was created; call submit again for this step "
                        "once capacity frees."
                    ),
                }

        # Reserve next: a crash after the AO call must not orphan a run, and a
        # concurrent process must not race us into a second launch.
        existing, created = self._store.reserve(candidate)
        if not created:
            # This step already holds a durable slot from its first launch, so
            # nothing is claimed again and no second run can start.
            return await self._reused_submit(existing, candidate, request)

        payload_request = effective_request(
            parent_goal_id, parent_step_id, request, request_id
        )
        try:
            cli_result = await self._client.submit(payload_request)
        except CliInvocationError as exc:
            # The reservation stays: the request_id is claimed, so a retry
            # reconnects rather than launching a duplicate. The caller is told
            # the launch outcome is unknown rather than being told it failed
            # the run.
            return _launch_unconfirmed(existing, exc)

        if not cli_result.ok:
            # AO refused the submission. A refusal alone does not prove that no
            # run exists, so the slot is released only once AO's own status
            # confirms the request is missing; otherwise the slot stays held
            # until a terminal status or result settles it.
            released = await self._release_after_confirmed_missing(existing)
            return {
                "ok": False,
                "outcome": "launch_failed",
                "request_id": existing.request_id,
                "parent_goal_id": parent_goal_id,
                "parent_step_id": parent_step_id,
                "run_id": None,
                "capacity_released": released,
                "error": (
                    f"AO CLI 'code-run-submit' exited {cli_result.exit_code}: "
                    f"{cli_error_detail(cli_result.stdout, cli_result.stderr)[:400]}"
                ),
                "detail": (
                    "The AO CLI refused the submission. The request_id stays "
                    "claimed so a later attempt reconnects instead of launching "
                    "a duplicate; read status or result on this request_id."
                ),
            }

        try:
            payload = parse_response(cli_result.stdout)
            shaped = validate_status_shape(payload)
        except MalformedResponseError as exc:
            return _launch_unconfirmed(
                existing,
                RuntimeError(
                    f"AO CLI 'code-run-submit' returned an unreadable response: {exc}"
                ),
            )

        if shaped["request_id"] != request_id:
            return _launch_unconfirmed(
                existing,
                MalformedResponseError(
                    f"AO submit answered for request_id {shaped['request_id']!r}, "
                    f"not {request_id!r}"
                ),
            )

        # AO's submit answers with the *durable state record* keyed by
        # request_id. It carries no CodeRunResult.run_id, and ao_run_dir is null
        # while the launch is still starting -- neither is a failed submission,
        # so neither may be treated as one. The run is controlled from here on
        # by request_id alone.
        state = self._store.update(request_id, state=STATE_SUBMITTED, submitted_at=_stamp())
        payload_out = _submit_payload(state or existing, launched=True)
        payload_out["ao_status"] = payload
        payload_out["execution_state"] = shaped["execution_state"]
        payload_out["execution_status"] = shaped["execution_status"]
        payload_out["change_acceptance"] = shaped["change_acceptance"]
        payload_out["ao_run_dir"] = shaped["ao_run_dir"]
        payload_out["result_available"] = shaped["result_available"]
        payload_out["lifecycle"] = lifecycle(
            shaped["execution_state"],
            shaped["execution_status"],
            shaped["change_acceptance"],
        )
        payload_out["run_id"] = self._record_run_id(state or existing, payload)
        return payload_out

    def _record_run_id(self, mapping: RunMapping, payload: dict[str, Any]) -> str | None:
        """Persist an AO ``run_id`` when AO has published its run identity.

        Submit and live status records identify an initialized AO run through
        ``ao_run_dir``; its basename is the same identifier used by the formal
        result. A null directory means dispatch has not published a run yet, so
        no identifier is inferred in that state.
        """
        run_id = payload.get("run_id")
        if not (isinstance(run_id, str) and run_id.strip()):
            ao_run_dir = payload.get("ao_run_dir")
            if isinstance(ao_run_dir, str) and ao_run_dir.strip():
                candidate = Path(ao_run_dir).name
                if candidate and candidate not in {".", ".."}:
                    run_id = candidate
        if isinstance(run_id, str) and run_id.strip():
            updated = self._store.update(mapping.request_id, run_id=run_id)
            return (updated or mapping).run_id
        return mapping.run_id

    async def _reused_submit(
        self, existing: RunMapping, candidate: RunMapping, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Reconcile an uncertain reservation, otherwise reuse its AO run.

        Raven writes the mapping before calling AO so a process crash cannot
        orphan a successfully launched run. A mapping still in ``reserved``
        means the process may have died either side of the CLI call. Ask AO
        first; only an explicit ``CodeRunNotFound`` permits a retry. AO's
        request-id reservation makes concurrent retries safe: one may win and
        the others reconcile through status.
        """
        payload = _submit_payload(existing, launched=False)
        if existing.request_fingerprint != candidate.request_fingerprint:
            # A different request for an identity that already owns a CodeRun is
            # a conflict, not a reuse. Reusing would run the wrong work under the
            # right request_id; relaunching would fork the run. Both are
            # refused, and the caller is told exactly what the stored
            # fingerprint was so it can decide.
            return {
                **payload,
                "ok": False,
                "outcome": "request_changed",
                "conflict": True,
                "request_changed": True,
                "stored_fingerprint": existing.request_fingerprint,
                "submitted_fingerprint": candidate.request_fingerprint,
                "detail": (
                    "An existing CodeRun is mapped to this parent step with a "
                    "different request. It was neither reused nor relaunched: "
                    "restore the original request, or use a new parent step."
                ),
            }
        if existing.state == STATE_STALE_MAPPING:
            # AO has confirmed this request no longer exists. That is reported,
            # not repaired: relaunching a forgotten run is the caller's call.
            return {
                **payload,
                "ok": False,
                "outcome": "stale_mapping",
                "run_not_found": True,
                "detail": (
                    "AO has no record of this request_id. The stale mapping is "
                    "kept and nothing was resubmitted; ask for a fresh parent step "
                    "or relaunch deliberately."
                ),
            }
        # Every pre-guard mapping needs a provisional slot before AO is queried.
        # If status is unavailable, retaining that slot prevents another submit
        # from exceeding the outer limit while this request remains uncertain.
        if not self._capacity.holds(existing.request_id):
            settled, overlay = await self._reconcile_resumed_capacity(existing, payload)
            if settled is not None:
                return settled
            payload.update(overlay)
            if overlay.get("capacity_adopted") is False:
                return payload
        if existing.cancel_requested:
            # A cancel is pending. Cancellation is never a reason to submit
            # again: the run is still AO's to settle, and the local marker
            # survives a restart precisely so this submit reconnects.
            return {
                **payload,
                "ok": True,
                "outcome": "cancel_pending",
                "cancel_requested": True,
                "detail": (
                    "A cancellation is pending on this request. Nothing was "
                    "submitted again; read status or result to wait for AO's "
                    "terminal cancelled result."
                ),
            }
        if existing.state == STATE_RESERVED and existing.submitted_at is None:
            try:
                status_result = await self._client.status(existing.request_id)
            except CliInvocationError as exc:
                return {
                    **_cli_failure(existing, "code-run-status", exc),
                    "capacity_provisional": self._capacity.holds(existing.request_id),
                }
            if status_result.ok:
                try:
                    status_payload = parse_response(status_result.stdout)
                    shaped = validate_status_shape(status_payload)
                    if shaped["request_id"] != existing.request_id:
                        raise MalformedResponseError("status answered for a different request_id")
                except MalformedResponseError as exc:
                    return {
                        **_cli_failure(existing, "code-run-status", exc),
                        "capacity_provisional": self._capacity.holds(existing.request_id),
                    }
                updated = self._store.update(
                    existing.request_id, state=STATE_SUBMITTED, submitted_at=_stamp()
                ) or existing
                recovered_run_id = self._record_run_id(updated, status_payload)
                recovered = _submit_payload(updated, launched=False)
                recovered.update(_submit_status_fields(status_payload, shaped))
                recovered["run_id"] = recovered_run_id
                recovered["outcome"] = "recovered"
                recovered["lifecycle"] = lifecycle(
                    shaped["execution_state"],
                    shaped["execution_status"],
                    shaped["change_acceptance"],
                )
                recovered["request_changed"] = payload.get("request_changed", False)
                if shaped["execution_state"] in _TERMINAL_STATES:
                    # AO's own status is the authority, and it says this run is
                    # already finished. A recovered reservation that AO has
                    # settled must not keep a live run's outer capacity hostage.
                    recovered["capacity_released"] = self._release_capacity(
                        existing.request_id
                    )
                else:
                    recovered["capacity_provisional"] = self._capacity.holds(
                        existing.request_id
                    )
                return recovered
            if not _is_missing_request(status_result):
                failure = CliInvocationError(
                    "could not determine whether the reserved request reached AO: "
                    + cli_error_detail(status_result.stdout, status_result.stderr)[:400]
                )
                return {
                    **_cli_failure(existing, "code-run-status", failure),
                    "capacity_provisional": self._capacity.holds(existing.request_id),
                }

            if existing.request_fingerprint != candidate.request_fingerprint:
                payload.update(
                    ok=False,
                    outcome="request_changed",
                    detail=(
                        "The reserved request has no AO record, and this submit "
                        "does not match its original fingerprint. No request was "
                        "launched; restore the original request before recovery."
                    ),
                )
                return payload

            # AO has explicitly confirmed this request_id does not exist. An
            # existing Raven mapping is never auto-resubmitted after that: the
            # run is recorded durably as stale and the caller decides. Silently
            # relaunching here would let a submit that never reached AO look
            # exactly like a recovered one.
            return self._mark_stale(existing, "code-run-submit")
        return payload

    # -- mapping resolution ---------------------------------------------
    async def _reconcile_resumed_capacity(
        self, existing: RunMapping, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Settle a resumed run against AO before its capacity slot is adopted.

        Mappings persisted before the outer-concurrency guard existed carry no
        active marker, so reconnecting to one would leave a running CodeRun
        uncounted. Only AO knows whether the run still needs a slot, so AO is
        asked after a provisional marker is durably claimed. A live run keeps
        that slot, while a terminal or missing request is reported from AO's
        authoritative answer and the provisional slot is released.

        Returns the authoritative result to return in place of the plain reuse
        payload, plus the fields to merge into that payload when the run is live
        but no slot could be claimed.
        """
        limit = self._config.max_concurrent_code_runs
        acquired, active = self._capacity.acquire(existing.request_id, limit)
        if not acquired:
            return None, {
                "capacity_adopted": False,
                "capacity_provisional": False,
                "limit": limit,
                "active_code_runs": active,
                "capacity_detail": (
                    f"Raven's outer AO concurrency limit is {limit} and that many "
                    "CodeRuns are already active, so this existing mapping keeps "
                    "its identity without launching another run. Retry after a "
                    "slot is available to reconcile it with AO."
                ),
            }
        try:
            status_result = await self._client.status(existing.request_id)
        except CliInvocationError as exc:
            return {
                **_cli_failure(existing, "code-run-status", exc),
                "capacity_provisional": True,
            }, {}
        if not status_result.ok:
            if _is_missing_request(status_result):
                return self._mark_stale(existing, "code-run-submit"), {}
            failure = CliInvocationError(
                "could not reconcile existing request with AO status: "
                + cli_error_detail(status_result.stdout, status_result.stderr)[:400]
            )
            return {
                **_cli_failure(existing, "code-run-status", failure),
                "capacity_provisional": True,
            }, {}
        try:
            status_body = parse_response(status_result.stdout)
            shaped = validate_status_shape(status_body)
            if shaped["request_id"] != existing.request_id:
                raise MalformedResponseError("status answered for a different request_id")
        except MalformedResponseError:
            error = MalformedResponseError(
                "AO status response was malformed or answered for a different request_id"
            )
            return {
                **_cli_failure(existing, "code-run-status", error),
                "capacity_provisional": True,
            }, {}
        if shaped["execution_state"] in _TERMINAL_STATES:
            settled_run_id = self._record_run_id(
                self._store.get(existing.request_id) or existing, status_body
            )
            return (
                {
                    **payload,
                    **_submit_status_fields(status_body, shaped),
                    "run_id": settled_run_id,
                    "outcome": "reconciled",
                    "capacity_adopted": False,
                    "capacity_released": self._release_capacity(existing.request_id),
                    "lifecycle": lifecycle(
                        shaped["execution_state"],
                        shaped["execution_status"],
                        shaped["change_acceptance"],
                    ),
                    "detail": (
                        f"AO reports this resumed request as {shaped['execution_state']}. "
                        "It was settled against AO without a relaunch, and its outer-AO "
                        "capacity slot was released."
                    ),
                },
                {},
            )
        self._record_run_id(
            self._store.get(existing.request_id) or existing, status_body
        )
        if existing.state == STATE_RESERVED and existing.submitted_at is None:
            updated = self._store.update(
                existing.request_id, state=STATE_SUBMITTED, submitted_at=_stamp()
            ) or existing
            recovered = _submit_payload(updated, launched=False)
            recovered.update(_submit_status_fields(status_body, shaped))
            recovered["run_id"] = updated.run_id
            recovered["outcome"] = "recovered"
            recovered["capacity_provisional"] = True
            recovered["lifecycle"] = lifecycle(
                shaped["execution_state"],
                shaped["execution_status"],
                shaped["change_acceptance"],
            )
            return recovered, {}
        return None, {
            "capacity_adopted": True,
            "capacity_provisional": False,
        }

    def _resolve(self, request_id: str | None, run_id: str | None) -> RunMapping:
        """Find the mapping that controls a run, by either identifier.

        ``run_id`` is accepted only as a convenience for a caller that captured
        one; it is looked up to recover the ``request_id`` AO is actually driven
        with, so a run whose ``run_id`` AO has not published yet is still
        fully controllable by its request id.
        """
        if request_id:
            mapping = self._store.get(request_id)
            if mapping is None:
                raise AdapterError(f"no mapping recorded for request_id {request_id!r}")
            return mapping
        if run_id:
            for candidate in self._iter_mappings():
                if candidate.run_id == run_id:
                    return candidate
            raise AdapterError(f"no Raven mapping found for run_id {run_id!r}")
        raise AdapterError("request_id or run_id is required")

    def _mark_stale(self, mapping: RunMapping, command: str) -> dict[str, Any]:
        """Record a confirmed-missing AO request and free its slot.

        The stale state is written durably so a later submit or status reports
        the same condition instead of quietly relaunching, and the outer-AO slot
        is released because AO has authoritatively confirmed the run is gone.
        """
        self._store.update(mapping.request_id, state=STATE_STALE_MAPPING)
        self._release_capacity(mapping.request_id)
        return _stale_mapping(mapping, command)

    def _iter_mappings(self):
        root = self._store.root
        if not root.exists():
            return
        for path in sorted(root.glob("*.json")):
            mapping = self._store.get(path.stem)
            if mapping is not None:
                yield mapping

    # -- events ---------------------------------------------------------
    async def events(
        self,
        request_id: str | None = None,
        run_id: str | None = None,
        after_seq: int | None = None,
    ) -> dict[str, Any]:
        mapping = self._resolve(request_id, run_id)
        # Serialize the complete read/fetch/advance operation across adapter
        # instances and processes. Advancing the cursor alone is not enough:
        # without this lock, a losing reader could still return the page it
        # fetched before another reader advanced the durable cursor.
        async with self._control_lock(mapping.request_id):
            mapping = self._store.get(mapping.request_id) or mapping
            return await self._events_locked(mapping, after_seq)

    async def _events_locked(
        self, mapping: RunMapping, after_seq: int | None
    ) -> dict[str, Any]:
        """Consume one incremental page of AO's durable event log.

        The cursor is the contract. ``after_seq`` defaults to the mapping's
        stored ``last_seen_event_seq``, so a process that restarts continues
        from exactly where it stopped: AO is asked for sequences *strictly
        greater* than the cursor, nothing is replayed, and nothing is skipped.
        An explicit ``after_seq`` can only move the reader forward -- asking for
        a range the cursor has already passed is refused rather than rewinding,
        so a stale caller cannot make two Raven processes fight over the same
        range.

        The page is validated before the cursor moves: request identity,
        per-event sequence monotonicity, timestamp shape, and event type. A
        page that fails validation advances nothing, so the next call re-reads
        the same range from a still-correct cursor.

        Nothing here reads a verdict. An event is something AO published, and
        the response reports events as a projection; acceptance and terminal
        state still come from AO's formal status and result.
        """
        cursor = mapping.last_seen_event_seq
        requested = cursor if after_seq is None else after_seq
        if (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested < CURSOR_BEFORE_FIRST_EVENT
        ):
            return _events_invalid(
                mapping,
                f"after_seq must be an integer of at least {CURSOR_BEFORE_FIRST_EVENT}, "
                f"got {after_seq!r}",
            )
        if requested < cursor:
            return _events_invalid(
                mapping,
                f"after_seq {requested} is behind the stored cursor {cursor}; "
                "the consumer cursor never moves backwards",
                after_seq=requested,
                cursor=cursor,
            )

        try:
            event_call = await self._client.events(mapping.request_id, requested)
        except (CliInvocationError, MalformedResponseError) as exc:
            # An infrastructure failure leaves the cursor exactly where it was,
            # so the same range is retried rather than lost. AO not knowing the
            # request is a different condition entirely and is reported as such.
            if isinstance(exc, CliInvocationError):
                return {
                    **_events_payload(mapping, requested, [], cursor),
                    **_cli_failure(mapping, "code-run-events", exc),
                    "detail": (
                        "The AO event page could not be read, so no cursor was "
                        "advanced and the same range is read again next time. "
                        "This is an invocation failure, not a verdict about the run."
                    ),
                }
            return {
                **_events_payload(mapping, requested, [], cursor),
                "ok": False,
                "outcome": "malformed_response",
                "error": str(exc),
                "detail": (
                    "The AO event page was not readable, so no cursor was "
                    "advanced and the same range is read again next time."
                ),
            }

        if _is_missing_request(event_call):
            stale = self._mark_stale(mapping, "code-run-events")
            return {
                **_events_payload(mapping, requested, [], cursor),
                **stale,
                "after_seq": requested,
                "events": [],
                "previous_cursor": cursor,
                "last_seen_event_seq": cursor,
                "cursor_advanced": False,
                "event_count": 0,
                "event_types": [],
            }

        try:
            payload = _formal(event_call, "code-run-events")
        except CliInvocationError as exc:
            return {
                **_events_payload(mapping, requested, [], cursor),
                **_cli_failure(mapping, "code-run-events", exc),
                "detail": (
                    "The AO event page could not be read, so no cursor was "
                    "advanced and the same range is read again next time. "
                    "This is an invocation failure, not a verdict about the run."
                ),
            }
        except MalformedResponseError as exc:
            return {
                **_events_payload(mapping, requested, [], cursor),
                "ok": False,
                "outcome": "malformed_response",
                "error": str(exc),
                "detail": (
                    "The AO event page was not readable, so no cursor was "
                    "advanced and the same range is read again next time."
                ),
            }

        try:
            page = _validate_event_page(payload, mapping, after_seq=requested)
        except MalformedResponseError as exc:
            return {
                **_events_payload(mapping, requested, [], cursor),
                "ok": False,
                "outcome": "malformed_response",
                "error": str(exc),
                "detail": (
                    "The AO event page failed validation, so no cursor was "
                    "advanced. Events are never partially consumed."
                ),
            }

        # Cursor first, then report: the durable position is the truth a
        # restart resumes from, so it is written before anything is returned.
        # A run_id AO published for the first time on this page is persisted in
        # the same locked write, not as a second update, so a crash can never
        # leave a cursor belonging to no run or a run whose cursor never moved.
        published_run_id = page.get("published_run_id")
        advanced = self._store.advance_event_cursor(
            mapping.request_id, page["high_watermark"], run_id=published_run_id
        )
        new_cursor = (advanced or mapping).last_seen_event_seq
        known_run_id = (advanced or mapping).run_id or mapping.run_id
        response = _events_payload(mapping, requested, page["events"], cursor)
        response["run_id"] = known_run_id
        response.update(
            {
                "ok": True,
                "outcome": "ok",
                "cursor_advanced": new_cursor > cursor,
                "last_seen_event_seq": new_cursor,
                "event_count": len(page["events"]),
                "event_types": page["event_types"],
            }
        )
        # A published run_id is a label, not a state change, and an event page
        # is not an authoritative terminal answer: capacity is untouched here.
        return response

    # -- status ---------------------------------------------------------
    async def status(
        self, request_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """Return AO's durable status, driven by ``request_id``."""
        mapping = self._resolve(request_id, run_id)
        try:
            status_call = await self._client.status(mapping.request_id)
        except CliInvocationError as exc:
            return _cli_failure(mapping, "code-run-status", exc)
        if _is_missing_request(status_call):
            return self._mark_stale(mapping, "code-run-status")
        try:
            parsed = _formal(status_call, "code-run-status")
        except MalformedResponseError as exc:
            return _cli_failure(mapping, "code-run-status", exc)
        if _is_interrupted(parsed):
            if parsed.get("request_id") != mapping.request_id:
                return _cli_failure(
                    mapping,
                    "code-run-status",
                    MalformedResponseError(
                        "status answered for a different request_id"
                    ),
                )
            # Observable and recoverable, not terminal: the slot stays held and
            # no acceptance is inferred from the interruption.
            return _interrupted_response(mapping, parsed, "code-run-status")
        try:
            payload = parsed
            shaped = validate_status_shape(payload)
            if shaped["request_id"] != mapping.request_id:
                raise MalformedResponseError("status answered for a different request_id")
        except MalformedResponseError as exc:
            return _cli_failure(mapping, "code-run-status", exc)

        run_id = self._record_run_id(mapping, payload)
        if shaped["execution_state"] in _TERMINAL_STATES:
            self._release_capacity(mapping.request_id)
        return {
            "ok": True,
            "outcome": "ok",
            "request_id": mapping.request_id,
            "run_id": run_id,
            "parent_goal_id": mapping.parent_goal_id,
            "parent_step_id": mapping.parent_step_id,
            "cancel_requested": mapping.cancel_requested,
            "source": "ao",
            # AO's own durable fields, passed through untouched.
            "ao_status": payload,
            "execution_state": shaped["execution_state"],
            "execution_status": shaped["execution_status"],
            "change_acceptance": shaped["change_acceptance"],
            "failure_class": shaped["failure_class"],
            "failure_reason": shaped["failure_reason"],
            "next_action": shaped["next_action"],
            "result_available": shaped["result_available"],
            "ao_run_dir": shaped["ao_run_dir"],
            "terminal": shaped["execution_state"] in _TERMINAL_STATES,
            "lifecycle": lifecycle(
                shaped["execution_state"],
                shaped["execution_status"],
                shaped["change_acceptance"],
            ),
        }

    # -- result ---------------------------------------------------------
    async def result(
        self, request_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """Return a formal CodeRunResult, preserving AO's verdict verbatim.

        The result is validated as a formal ``CodeRunResult`` *and* checked
        against the mapping it was fetched for: a payload answering for a
        different ``request_id`` is a contract violation, not a result for this
        step.

        ``accepted_commit`` / ``accepted_patch`` appear only under AO's own
        ``accepted`` acceptance with a concrete artifact and
        ``verification_complete`` true. Every other case reports no accepted
        artifact, including a run whose ``execution_status`` is ``completed``.
        """
        mapping = self._resolve(request_id, run_id)
        try:
            result_call = await self._client.result(mapping.request_id)
        except CliInvocationError as exc:
            return _cli_failure(mapping, "code-run-result", exc)
        if _is_missing_request(result_call):
            return self._mark_stale(mapping, "code-run-result")
        try:
            payload = _formal(result_call, "code-run-result")
            shaped = validate_result_shape(payload)
        except MalformedResponseError as exc:
            return _cli_failure(mapping, "code-run-result", exc)

        if shaped["request_id"] != mapping.request_id:
            return {
                "ok": False,
                "outcome": "malformed_response",
                "source": "ao",
                "request_id": mapping.request_id,
                "command": "code-run-result",
                "error": (
                    f"result answers for request_id {shaped['request_id']!r}, "
                    f"not {mapping.request_id!r}"
                ),
                "detail": (
                    "The AO result does not belong to this Raven step, so no "
                    "verdict from it is reported here."
                ),
            }
        if (
            shaped["parent_goal_id"] != mapping.parent_goal_id
            or shaped["parent_step_id"] != mapping.parent_step_id
        ):
            return {
                "ok": False,
                "outcome": "malformed_response",
                "source": "ao",
                "request_id": mapping.request_id,
                "command": "code-run-result",
                "error": "result parent identifiers do not match the Raven mapping",
                "detail": "The AO result does not belong to this Raven goal and step.",
            }

        # AO names the run on the result; keep it for a caller that reconnects
        # by run_id, but the request_id remains the control key.
        self._record_run_id(mapping, payload)

        accepted_artifact = shaped["accepted_artifact"]
        raven_lifecycle = lifecycle(
            None, shaped["execution_status"], shaped["change_acceptance"]
        )
        ao_result = dict(payload)
        if accepted_artifact is None:
            ao_result.pop("accepted_commit", None)
            ao_result.pop("accepted_patch", None)
            if shaped["change_acceptance"] == "accepted":
                raven_lifecycle = "acceptance_unverified"

        response: dict[str, Any] = {
            "ok": True,
            "outcome": "ok",
            "source": "ao",
            "request_id": mapping.request_id,
            "run_id": shaped["run_id"],
            "parent_goal_id": mapping.parent_goal_id,
            "parent_step_id": mapping.parent_step_id,
            # AO's result axis and acceptance axis stay separate fields.
            "execution_status": shaped["execution_status"],
            "execution_terminal": shaped["execution_terminal"],
            "change_acceptance": shaped["change_acceptance"],
            "failure_class": shaped["failure_class"],
            "failure_reason": shaped["failure_reason"],
            "next_action": shaped["next_action"],
            "verification_complete": shaped["verification_complete"],
            "ao_result": ao_result,
            "accepted_artifact_exposed": accepted_artifact is not None,
            "lifecycle": raven_lifecycle,
        }
        if accepted_artifact is not None:
            response.update(accepted_artifact)
        cancelled = shaped["execution_status"] == "cancelled"
        settled = shaped["execution_terminal"]
        if settled:
            self._release_capacity(mapping.request_id)
        self._settle_cancel(mapping, cancelled=cancelled, settled=settled)
        if mapping.cancel_requested and cancelled:
            response["cancel_confirmed"] = True
        return response

    def _settle_cancel(
        self, mapping: RunMapping, *, cancelled: bool, settled: bool
    ) -> None:
        """Retire the local cancel marker once the run can no longer change.

        A cancel *request* is not a cancellation: AO reports ``cancel_requested``
        until dispatcher termination is actually confirmed, so the marker is
        kept for the whole live window. It is retired when AO reaches a terminal
        state -- either the confirmed ``cancelled`` or a run that settled for
        some other reason, where asking again could not change the outcome and a
        permanently pending flag would misreport a finished run.

        Only ``cancelled`` counts as confirmation; a terminal state that is not a
        cancellation is never reported as one.
        """
        if not mapping.cancel_requested or not settled:
            return
        self._store.update(
            mapping.request_id,
            cancel_requested=False,
            state=STATE_SUBMITTED if cancelled else mapping.state,
        )

    # -- cancel ---------------------------------------------------------
    async def cancel(
        self,
        request_id: str | None = None,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Request cancellation and return AO's durable state.

        ``cancel_requested`` stays nonterminal in Raven. AO moves to
        ``cancel_requested`` the moment a stop is asked for and only reaches
        ``cancelled`` once dispatcher termination is confirmed, so the local
        record is marked nonterminal unless AO has *already* confirmed. Nothing
        is auto-retried here.
        """
        mapping = self._resolve(request_id, run_id)
        try:
            cancel_call = await self._client.cancel(mapping.request_id, reason)
        except CliInvocationError as exc:
            return _cli_failure(mapping, "code-run-cancel", exc)
        if _is_missing_request(cancel_call):
            return self._mark_stale(mapping, "code-run-cancel")
        try:
            parsed = _formal(cancel_call, "code-run-cancel")
        except MalformedResponseError as exc:
            return _cli_failure(mapping, "code-run-cancel", exc)
        if _is_interrupted(parsed):
            if parsed.get("request_id") != mapping.request_id:
                return _cli_failure(
                    mapping,
                    "code-run-cancel",
                    MalformedResponseError(
                        "cancel answered for a different request_id"
                    ),
                )
            return _interrupted_response(mapping, parsed, "code-run-cancel")
        try:
            payload = parsed
            shaped = validate_status_shape(payload)
            if shaped["request_id"] != mapping.request_id:
                raise MalformedResponseError("cancel answered for a different request_id")
        except MalformedResponseError as exc:
            return _cli_failure(mapping, "code-run-cancel", exc)

        # AO answers a cancel that lost the race with its own terminal verdict,
        # and a cancel asked for twice is a no-op that returns the same record.
        # Only a real 'cancelled' is a confirmation; a terminal state that is not
        # a cancellation means the run settled first and the stop is moot.
        execution_state = shaped["execution_state"]
        confirmed = execution_state == "cancelled"
        settled = execution_state in _TERMINAL_STATES
        if confirmed:
            outcome = "cancelled"
        elif settled:
            outcome = "moot"
        else:
            outcome = "cancel_requested"
        self._settle_cancel(mapping, cancelled=confirmed, settled=settled)
        if settled:
            # AO reported a terminal state, which is the only thing that frees
            # the outer-AO slot. A pending cancel keeps holding it.
            self._release_capacity(mapping.request_id)
        pending = not settled
        updated = self._store.update(
            mapping.request_id,
            cancel_requested=pending,
            state=STATE_CANCEL_REQUESTED if pending else STATE_SUBMITTED,
        )
        return {
            "ok": True,
            "outcome": "ok",
            "source": "ao",
            "request_id": mapping.request_id,
            "run_id": mapping.run_id,
            "ao_status": payload,
            "execution_state": execution_state,
            "execution_status": shaped["execution_status"],
            "change_acceptance": shaped["change_acceptance"],
            "failure_class": shaped["failure_class"],
            "next_action": shaped["next_action"],
            "cancel_outcome": outcome,
            "cancel_requested": pending,
            "cancel_confirmed": confirmed,
            "terminal": settled,
            "raven_state": (updated or mapping).state,
            "lifecycle": lifecycle(
                execution_state,
                shaped["execution_status"],
                shaped["change_acceptance"],
            ),
        }


_TERMINAL_STATES = frozenset({"terminal", "blocked", "failed", "cancelled"})


def _interrupted_response(
    mapping: RunMapping, payload: dict[str, Any], command: str
) -> dict[str, Any]:
    """Report an AO ``interrupted`` run as its own observable state.

    AO can report a run it stopped rather than finished. That is neither a
    terminal verdict nor a failure, and it is not ``cancel_requested`` either,
    so it is surfaced as ``lifecycle="interrupted"`` with ``terminal`` false and
    ``recoverable`` true. The run keeps its outer-AO slot: an interrupted run
    may still be resumed, and only AO can say it is done.

    This is read before the closed status shape is validated, because the
    adapter treats an unknown-but-reported lifecycle word as a contract change
    to surface, not as a reason to report the whole response as unreadable.
    """
    return {
        "ok": True,
        "outcome": "interrupted",
        "source": "ao",
        "request_id": mapping.request_id,
        "run_id": mapping.run_id,
        "parent_goal_id": mapping.parent_goal_id,
        "parent_step_id": mapping.parent_step_id,
        "command": command,
        "cancel_requested": mapping.cancel_requested,
        "execution_state": "interrupted",
        "lifecycle": "interrupted",
        "interrupted": True,
        "recoverable": True,
        "terminal": False,
        "result_available": bool(payload.get("result_available")),
        "ao_status": payload,
        "detail": (
            "AO reports this run as interrupted. It is not terminal and not a "
            "failure: the run keeps its concurrency slot and stays resumable. "
            "No acceptance value is available until AO publishes a result."
        ),
    }


def _is_interrupted(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and payload.get("execution_state") == "interrupted"


def _validate_event_page(
    payload: dict[str, Any], mapping: RunMapping, *, after_seq: int
) -> dict[str, Any]:
    """Gate one AO event page before the cursor is allowed to move.

    The page is read as *data*: AO's own ``request_id`` must match the mapping
    being read (a page for another run is never consumed), every event must
    carry an integer ``sequence`` strictly greater than the requested cursor and
    strictly greater than its predecessor inside the page, every timestamp must
    parse as an ISO-8601 date and time, and every event type must be a non-empty
    string. AO's own timestamp string is carried through unchanged.

    Every event must carry a non-empty string ``run_id``: AO stamps each event
    it publishes with the run it belongs to, so an event without one would
    advance the cursor on evidence of an unidentified run. When the mapping
    already knows a run, the event's ``run_id`` must match it, and it must match
    the page's own ``run_id`` whenever the page reports one. A page emitted
    before AO published a run identity is still valid as long as it is empty,
    and a ``run_id`` AO publishes for the first time on such a page is returned
    so the caller can persist it together with the cursor.
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError("event page was not a JSON object")
    page_request_id = payload.get("request_id")
    if page_request_id != mapping.request_id:
        raise MalformedResponseError(
            f"event page answers for request_id {page_request_id!r}, "
            f"not {mapping.request_id!r}"
        )
    page_run_id = payload.get("run_id")
    if not isinstance(page_run_id, (str, type(None))):
        raise MalformedResponseError("event page field 'run_id' was not a string")
    if page_run_id is not None and mapping.run_id is not None and page_run_id != mapping.run_id:
        raise MalformedResponseError(
            f"event page answers for run_id {page_run_id!r}, not {mapping.run_id!r}"
        )
    events = payload.get("events")
    if not isinstance(events, list):
        raise MalformedResponseError("event page field 'events' was not a list")

    previous = after_seq
    validated: list[dict[str, Any]] = []
    published_run_id: str | None = None
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise MalformedResponseError(f"event {index} was not a JSON object")
        event_request_id = raw.get("request_id")
        if event_request_id is not None and event_request_id != mapping.request_id:
            raise MalformedResponseError(
                f"event {index} carries request_id {event_request_id!r}, "
                f"not {mapping.request_id!r}"
            )
        # AO's accepted field name is 'sequence'; 'seq' is accepted as a legacy
        # alias so an older page shape still reads, never as a silent default.
        sequence = raw.get("sequence", raw.get("seq"))
        if sequence is None:
            raise MalformedResponseError(
                f"event {index} field 'sequence' was not an integer"
            )
        if isinstance(sequence, bool) or not isinstance(sequence, (int, str)):
            raise MalformedResponseError(f"event {index} field 'sequence' was not an integer")
        if isinstance(sequence, str):
            if not _SEQ_RE.match(sequence):
                raise MalformedResponseError(
                    f"event {index} field 'sequence' was not a decimal sequence"
                )
            sequence = int(sequence)
        if sequence <= previous:
            raise MalformedResponseError(
                f"event {index} sequence {sequence} is not strictly greater "
                f"than {previous}"
            )
        previous = sequence

        # AO stamps every event it publishes with the run it belongs to, so an
        # event reaching Raven without one is a page shape Raven cannot trust:
        # it would advance a cursor on evidence of an unidentified run. The
        # check is per event, so an empty page still reads before AO has
        # published any run identity at all.
        event_run_id = raw.get("run_id")
        if not isinstance(event_run_id, str) or not event_run_id.strip():
            raise MalformedResponseError(
                f"event {index} field 'run_id' was not a non-empty string"
            )
        if mapping.run_id is not None:
            if event_run_id != mapping.run_id:
                raise MalformedResponseError(
                    f"event {index} carries run_id {event_run_id!r}, "
                    f"not {mapping.run_id!r}"
                )
        if page_run_id is not None:
            # An event must belong to the very page that carried it. A page
            # labelled with one run and holding another run's events is a mixed
            # response, and reading it would attribute one run's progress to
            # another's cursor.
            if event_run_id != page_run_id:
                raise MalformedResponseError(
                    f"event {index} carries run_id {event_run_id!r}, "
                    f"but the page answers for {page_run_id!r}"
                )
        if published_run_id is None:
            published_run_id = event_run_id

        event_type = raw.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise MalformedResponseError(
                f"event {index} field 'event_type' was not a non-empty string"
            )
        timestamp = _parse_event_timestamp(raw.get("timestamp"))
        validated.append(
            {
                "sequence": sequence,
                "event_type": event_type,
                "timestamp": timestamp,
                "run_id": event_run_id,
                "detail": raw.get("detail") if isinstance(raw.get("detail"), (str, dict)) else None,
            }
        )
    if published_run_id is None and isinstance(page_run_id, str) and page_run_id.strip():
        published_run_id = page_run_id
    return {
        "events": validated,
        "high_watermark": previous,
        "published_run_id": published_run_id,
        "event_types": [event["event_type"] for event in validated],
    }


def _events_payload(
    mapping: RunMapping,
    after_seq: int,
    events: list[dict[str, Any]],
    cursor: int,
) -> dict[str, Any]:
    return {
        "source": "ao_events",
        "request_id": mapping.request_id,
        "run_id": mapping.run_id,
        "parent_goal_id": mapping.parent_goal_id,
        "parent_step_id": mapping.parent_step_id,
        "after_seq": after_seq,
        "events": events,
        "previous_cursor": cursor,
    }


def _events_invalid(
    mapping: RunMapping, error: str, *, after_seq: int | None = None, cursor: int | None = None
) -> dict[str, Any]:
    return {
        **_events_payload(
            mapping,
            mapping.last_seen_event_seq if after_seq is None else after_seq,
            [],
            mapping.last_seen_event_seq if cursor is None else cursor,
        ),
        "ok": False,
        "outcome": "invalid_request",
        "error": error,
        "detail": "No events were read and no cursor was advanced.",
    }


def _stale_mapping(mapping: RunMapping, command: str) -> dict[str, Any]:
    """Report a mapping whose AO request no longer exists, without relaunching.

    This is distinct from an infrastructure failure: AO answered, and its answer
    was that it has no record of this request. The condition is recorded durably
    so it stays observable across a restart, and the outer-AO slot is released
    because AO has confirmed the run is not there. No acceptance value is
    invented, and nothing is resubmitted -- only the caller can decide whether a
    missing run should be replaced.
    """
    return {
        "ok": False,
        "outcome": "stale_mapping",
        "source": "ao",
        "request_id": mapping.request_id,
        "run_id": mapping.run_id,
        "parent_goal_id": mapping.parent_goal_id,
        "parent_step_id": mapping.parent_step_id,
        "command": command,
        "run_not_found": True,
        "raven_state": STATE_STALE_MAPPING,
        "error": (
            f"AO has no record of request_id {mapping.request_id!r}; the Raven "
            "mapping now points at a run that does not exist"
        ),
        "detail": (
            "The mapping is kept and marked stale. Nothing was resubmitted: "
            "deciding whether to relaunch a forgotten run is the caller's call."
        ),
    }


def _is_missing_request(result: CliResult) -> bool:
    """Whether AO explicitly confirmed that no durable request exists."""
    if result.ok:
        return False
    try:
        payload = parse_response(result.stdout)
    except MalformedResponseError:
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("type") == "CodeRunNotFound"


def _submit_status_fields(
    payload: dict[str, Any], shaped: dict[str, Any]
) -> dict[str, Any]:
    """Map a durable AO status onto the submit/recovery response."""
    return {
        "ao_status": payload,
        "execution_state": shaped["execution_state"],
        "execution_status": shaped["execution_status"],
        "change_acceptance": shaped["change_acceptance"],
        "ao_run_dir": shaped["ao_run_dir"],
        "result_available": shaped["result_available"],
    }


def _launch_unconfirmed(mapping: RunMapping, exc: Exception) -> dict[str, Any]:
    """Report an unconfirmed launch without claiming the run failed.

    The mapping is retained either way, so the next submit or status reconnects
    to the same ``request_id`` instead of launching a second CodeRun.
    """
    return {
        "ok": False,
        "outcome": "launch_unconfirmed",
        "request_id": mapping.request_id,
        "parent_goal_id": mapping.parent_goal_id,
        "parent_step_id": mapping.parent_step_id,
        "run_id": mapping.run_id,
        "error": str(exc),
        "detail": (
            "Submission could not be confirmed. The mapping is retained, "
            "so a later submit or status reconnects to this request_id "
            "instead of launching another CodeRun."
        ),
    }


def _submit_payload(mapping: RunMapping, launched: bool) -> dict[str, Any]:
    return {
        "ok": True,
        "outcome": "submitted" if launched else "reused",
        "launched": launched,
        "source": "raven_mapping",
        "request_id": mapping.request_id,
        "parent_goal_id": mapping.parent_goal_id,
        "parent_step_id": mapping.parent_step_id,
        "run_id": mapping.run_id,
        "state": mapping.state,
        "request_fingerprint": mapping.request_fingerprint,
        "detail": (
            None
            if launched
            else "Reused the existing CodeRun mapped to this parent step; "
            "no duplicate run was launched."
        ),
    }


def _cli_failure(mapping: RunMapping, command: str, exc: Exception) -> dict[str, Any]:
    """Report a CLI failure without inventing a verdict.

    ``change_acceptance`` is deliberately absent: no formal AO response was
    read, so no acceptance value is known.
    """
    return {
        "ok": False,
        "outcome": "cli_failed" if isinstance(exc, CliInvocationError) else "malformed_response",
        "source": "ao_cli",
        "request_id": mapping.request_id,
        "run_id": mapping.run_id,
        "command": command,
        "error": str(exc),
        "detail": (
            "The AO CLI did not return a formal response, so no acceptance "
            "value is available. This is not a verdict about the run."
        ),
    }


__all__ = [
    "ACCEPTANCE_ACCEPTED",
    "AdapterError",
    "CodeRunAdapter",
    "effective_request",
]
