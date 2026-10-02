"""Tests for Raven's durable adapter over the AO CodeRun JSON contract."""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from raven_ao.adapter import CodeRunAdapter
from raven_ao.client import (
    AO_COMMANDS,
    COMMAND_CANCEL,
    COMMAND_EVENTS,
    COMMAND_RESULT,
    COMMAND_STATUS,
    COMMAND_SUBMIT,
    AoCliClient,
    CliInvocationError,
    CliResult,
)
from raven_ao.config import AdapterConfig
from raven_ao.contract import MalformedResponseError, validate_result_shape
from raven_ao.mapping import MappingCorruptError, MappingStore, RunMapping, derive_request_id, fingerprint_request
from raven_ao.tools import (
    CancelTool,
    EventsTool,
    ResultTool,
    StatusTool,
    SubmitTool,
    make_cancel_tool,
    make_events_tool,
    make_result_tool,
    make_status_tool,
    make_submit_tool,
)

GOAL = "goal-7"
STEP = "step-3"
REQUEST: dict[str, Any] = {
    "schema_version": "1.0",
    "repo": "/repo",
    "base_commit": "a" * 40,
    "package_id": "package-1",
    "tasks": [{"id": "T001", "title": "Change one file", "worker": "minimax"}],
    "acceptance_criteria": ["focused tests pass"],
}


def status_payload(
    request_id: str,
    *,
    execution_state: str = "running",
    execution_status: str = "running",
    change_acceptance: str = "not_evaluated",
    failure_class: str = "none",
    failure_reason: str = "",
    result_available: bool = False,
    ao_run_dir: str | None = "/tmp/ao-runs/run-1",
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "repo": "/repo",
        "base_commit": "a" * 40,
        "base_drift": None,
        "execution_state": execution_state,
        "execution_status": execution_status,
        "change_acceptance": change_acceptance,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "supervisor_pid": 123,
        "ao_run_dir": ao_run_dir,
        "created_at": "2026-10-02T00:00:00+00:00",
        "updated_at": "2026-10-02T00:01:00+00:00",
        "mode": "nonblocking",
        "result_available": result_available,
        "next_action": "none",
    }


def result_payload(
    request_id: str,
    *,
    run_id: str = "run-1",
    execution_status: str = "completed",
    change_acceptance: str = "accepted",
    failure_class: str = "none",
    failure_reason: str = "",
    verification_complete: bool = True,
    accepted_commit: str = "accepted-commit",
    accepted_patch: str = "diff --git a/file b/file\n",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "run_id": run_id,
        "execution_status": execution_status,
        "change_acceptance": change_acceptance,
        "base_commit": "a" * 40,
        "effective_base_commit": "a" * 40,
        "parent_goal_id": GOAL,
        "parent_step_id": STEP,
        "package_id": "package-1",
        "task_evidence": [],
        "attempt_evidence": [],
        "scope_evidence": {"status": "passed", "complete": True},
        "tests": [],
        "verification_complete": verification_complete,
        "review_evidence": {"status": "passed", "complete": True},
        "acceptance_evidence": {"status": "passed", "complete": True},
        "accepted_commit": accepted_commit,
        "accepted_patch": accepted_patch,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "next_action": "handoff_for_merge" if change_acceptance == "accepted" else "none",
        "started_at": "2026-10-02T00:00:00+00:00",
        "finished_at": "2026-10-02T00:01:00+00:00",
    }


class FakeClient:
    """AO boundary stand-in that emits its real JSON shapes."""

    def __init__(self) -> None:
        self.submits: list[dict[str, Any]] = []
        self.statuses: list[str] = []
        self.results: list[str] = []
        self.cancels: list[tuple[str, str | None]] = []
        self.event_reads: list[tuple[str, int]] = []
        self.status_response: dict[str, Any] | None = None
        self.result_response: dict[str, Any] | None = None
        self.cancel_response: dict[str, Any] | None = None
        self.event_page: dict[str, Any] | None = None
        self.event_exit = 0
        self.events_error: Exception | None = None
        self.submit_exit = 0
        self.submit_stderr = ""
        self.status_exit = 0
        self.status_error: Exception | None = None

    async def submit(self, request: dict[str, Any]) -> CliResult:
        self.submits.append(request)
        payload = status_payload(request["request_id"], ao_run_dir=None)
        return CliResult(self.submit_exit, json.dumps(payload), self.submit_stderr)

    async def status(self, request_id: str) -> CliResult:
        self.statuses.append(request_id)
        if self.status_error is not None:
            raise self.status_error
        if self.status_exit:
            return CliResult(
                self.status_exit,
                json.dumps({"error": {"type": "CodeRunNotFound", "message": "missing"}}),
                "",
            )
        payload = self.status_response or status_payload(request_id)
        payload.setdefault("request_id", request_id)
        return CliResult(0, json.dumps(payload), "")

    async def result(self, request_id: str) -> CliResult:
        self.results.append(request_id)
        payload = self.result_response or result_payload(request_id)
        payload.setdefault("request_id", request_id)
        return CliResult(0, json.dumps(payload), "")

    async def cancel(self, request_id: str, reason: str | None = None) -> CliResult:
        self.cancels.append((request_id, reason))
        payload = self.cancel_response or status_payload(
            request_id,
            execution_state="cancel_requested",
            execution_status="running",
            change_acceptance="not_evaluated",
            result_available=False,
        )
        payload.setdefault("request_id", request_id)
        return CliResult(0, json.dumps(payload), "")

    async def events(self, request_id: str, after_seq: int = -1) -> CliResult:
        self.event_reads.append((request_id, after_seq))
        if self.events_error is not None:
            raise self.events_error
        if self.event_exit:
            return CliResult(
                self.event_exit,
                json.dumps({"error": {"type": "CodeRunNotFound", "message": "missing"}}),
                "",
            )
        payload = self.event_page if self.event_page is not None else {}
        payload = dict(payload)
        payload.setdefault("request_id", request_id)
        payload.setdefault("events", [])
        return CliResult(0, json.dumps(payload), "")


def event(sequence: int, event_type: str = "run_progress", **extra: Any) -> dict[str, Any]:
    # AO stamps every published event with the run it belongs to; the default
    # keeps a fixture event shaped like AO's own projection unless a test is
    # deliberately about the run identity.
    return {
        "sequence": sequence,
        "event_type": event_type,
        "timestamp": f"2026-10-02T00:00:{sequence:02d}Z",
        "run_id": "run-1",
        **extra,
    }


def event_page(request_id: str, events: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"request_id": request_id, "events": events, **extra}


def make_adapter(tmp_path: Path, client: FakeClient) -> CodeRunAdapter:
    config = AdapterConfig(
        executable=("python3", "-m", "orchestrator.cli"),
        runs_dir=tmp_path / "runs",
        mapping_dir=tmp_path / "mapping",
    )
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    return CodeRunAdapter(config, client=client)


async def submit_once(adapter: CodeRunAdapter) -> dict[str, Any]:
    return await adapter.submit(GOAL, STEP, REQUEST)


def test_request_id_is_stable_and_parent_derived() -> None:
    assert derive_request_id(GOAL, STEP) == derive_request_id(GOAL, STEP)
    assert derive_request_id(GOAL, STEP) != derive_request_id(GOAL, "step-4")
    assert derive_request_id("a", "bc") != derive_request_id("ab", "c")


def test_mapping_is_durable_and_contains_only_the_parent_link(tmp_path: Path) -> None:
    store = MappingStore(tmp_path / "mapping")
    mapping = RunMapping(
        request_id=derive_request_id(GOAL, STEP),
        parent_goal_id=GOAL,
        parent_step_id=STEP,
        request_fingerprint="fingerprint",
    )
    reserved, created = store.reserve(mapping)
    assert created
    store.update(reserved.request_id, run_id="run-1")
    recovered = MappingStore(tmp_path / "mapping").get(reserved.request_id)
    assert recovered is not None
    assert recovered.parent_goal_id == GOAL
    assert recovered.parent_step_id == STEP
    assert recovered.run_id == "run-1"
    persisted = json.loads((tmp_path / "mapping" / f"{reserved.request_id}.json").read_text())
    assert not {"tasks", "attempts", "execution_state", "execution_status", "dag"} & set(persisted)


async def test_submit_injects_stable_ao_and_parent_ids_and_keeps_live_axes(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    out = await submit_once(adapter)
    request_id = derive_request_id(GOAL, STEP)
    assert client.submits == [{
        **REQUEST,
        "request_id": request_id,
        "parent_goal_id": GOAL,
        "parent_step_id": STEP,
    }]
    assert out["request_id"] == request_id
    assert out["run_id"] is None
    assert out["execution_state"] == "running"
    assert out["change_acceptance"] == "not_evaluated"


async def test_accepted_result_exposes_only_ao_verified_artifact(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.result_response = result_payload(submitted["request_id"])
    out = await adapter.result(request_id=submitted["request_id"])
    assert out["execution_status"] == "completed"
    assert out["execution_terminal"] is True
    assert out["change_acceptance"] == "accepted"
    assert out["lifecycle"] == "accepted"
    assert out["accepted_artifact_exposed"] is True
    assert out["accepted_commit"] == "accepted-commit"
    assert out["accepted_patch"] == "diff --git a/file b/file\n"


@pytest.mark.parametrize(
    "overrides",
    [
        {"change_acceptance": "accepted", "verification_complete": False},
        {"change_acceptance": "accepted", "accepted_commit": "", "accepted_patch": ""},
        {"change_acceptance": "rejected", "failure_class": "scope_violation"},
    ],
)
async def test_unaccepted_or_unverified_result_hides_artifact(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.result_response = result_payload(submitted["request_id"], **overrides)
    out = await adapter.result(request_id=submitted["request_id"])
    assert out["accepted_artifact_exposed"] is False
    assert "accepted_commit" not in out
    assert "accepted_patch" not in out
    assert "accepted_commit" not in out["ao_result"]
    assert "accepted_patch" not in out["ao_result"]
    assert out["change_acceptance"] == overrides["change_acceptance"]
    if overrides["change_acceptance"] == "accepted":
        assert out["lifecycle"] == "acceptance_unverified"


async def test_scope_violation_rejected_even_after_completed_execution(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.result_response = result_payload(
        submitted["request_id"],
        execution_status="completed",
        change_acceptance="rejected",
        failure_class="scope_violation",
        failure_reason="touched a file outside the approved scope",
        accepted_commit="must-not-leak",
        accepted_patch="must-not-leak",
    )
    out = await adapter.result(request_id=submitted["request_id"])
    assert out["execution_status"] == "completed"
    assert out["lifecycle"] == "rejected"
    assert out["failure_class"] == "scope_violation"
    assert out["accepted_artifact_exposed"] is False
    assert "accepted_commit" not in out and "accepted_patch" not in out
    assert "accepted_commit" not in out["ao_result"]
    assert "accepted_patch" not in out["ao_result"]


@pytest.mark.parametrize(
    ("execution_status", "failure_class", "expected_lifecycle"),
    [
        ("blocked", "dependency_conflict", "blocked"),
        ("blocked", "base_drift", "blocked"),
        ("failed", "infrastructure", "failed"),
        ("failed", "verification", "failed"),
        ("blocked", "needs_input", "blocked"),
        ("cancelled", "cancellation", "cancelled"),
    ],
)
async def test_failure_classes_and_execution_axis_are_preserved(
    tmp_path: Path,
    execution_status: str,
    failure_class: str,
    expected_lifecycle: str,
) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.result_response = result_payload(
        submitted["request_id"],
        execution_status=execution_status,
        change_acceptance="pending",
        failure_class=failure_class,
        failure_reason=f"fixture: {failure_class}",
        accepted_commit="",
        accepted_patch="",
    )
    out = await adapter.result(request_id=submitted["request_id"])
    assert out["execution_status"] == execution_status
    assert out["change_acceptance"] == "pending"
    assert out["failure_class"] == failure_class
    assert out["failure_reason"] == f"fixture: {failure_class}"
    assert out["lifecycle"] == expected_lifecycle
    assert out["accepted_artifact_exposed"] is False


async def test_cancel_requested_is_nonterminal_until_formal_result_confirms(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    out = await adapter.cancel(request_id=submitted["request_id"], reason="superseded")
    assert client.cancels == [(submitted["request_id"], "superseded")]
    assert out["ok"] is True
    assert out["cancel_confirmed"] is False
    assert out["cancel_requested"] is True
    assert out["execution_state"] == "cancel_requested"
    assert out["terminal"] is False

    pending = MappingStore(tmp_path / "mapping").get(submitted["request_id"])
    assert pending is not None and pending.cancel_requested is True

    client.result_response = result_payload(
        submitted["request_id"],
        execution_status="cancelled",
        change_acceptance="pending",
        failure_class="cancellation",
        accepted_commit="",
        accepted_patch="",
    )
    confirmed = await adapter.result(request_id=submitted["request_id"])
    assert confirmed["change_acceptance"] == "pending"
    assert confirmed["execution_status"] == "cancelled"
    assert confirmed["failure_class"] == "cancellation"
    assert confirmed["cancel_confirmed"] is True
    retired = MappingStore(tmp_path / "mapping").get(submitted["request_id"])
    assert retired is not None and retired.cancel_requested is False


async def test_cancel_race_returns_ao_terminal_state_without_downgrading(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.cancel_response = status_payload(
        submitted["request_id"],
        execution_state="cancelled",
        execution_status="cancelled",
        change_acceptance="pending",
        failure_class="cancellation",
        result_available=True,
    )
    out = await adapter.cancel(request_id=submitted["request_id"])
    assert out["execution_state"] == "cancelled"
    assert out["cancel_confirmed"] is True
    assert out["cancel_requested"] is False


async def test_status_preserves_ao_durable_state_and_not_evaluated(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.status_response = status_payload(submitted["request_id"])
    out = await adapter.status(request_id=submitted["request_id"])
    assert out["source"] == "ao"
    assert out["execution_state"] == "running"
    assert out["execution_status"] == "running"
    assert out["change_acceptance"] == "not_evaluated"
    assert out["terminal"] is False
    assert out["ao_status"]["result_available"] is False
    assert out["run_id"] == "run-1"
    mapped = adapter.store.get(submitted["request_id"])
    assert mapped is not None and mapped.run_id == "run-1"


async def test_status_does_not_infer_run_id_before_ao_publishes_run_directory(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.status_response = status_payload(submitted["request_id"], ao_run_dir=None)
    out = await adapter.status(request_id=submitted["request_id"])
    assert out["run_id"] is None
    mapped = adapter.store.get(submitted["request_id"])
    assert mapped is not None and mapped.run_id is None


async def test_status_and_cancel_reject_responses_for_another_request(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.status_response = status_payload("wrong-request")
    status = await adapter.status(request_id=submitted["request_id"])
    assert status["ok"] is False and status["outcome"] == "malformed_response"
    client.cancel_response = status_payload("wrong-request")
    cancelled = await adapter.cancel(request_id=submitted["request_id"])
    assert cancelled["ok"] is False and cancelled["outcome"] == "malformed_response"


async def test_moot_cancel_clears_local_pending_state(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.cancel_response = status_payload(
        submitted["request_id"],
        execution_state="blocked",
        execution_status="blocked",
        change_acceptance="pending",
        failure_class="dependency_conflict",
        result_available=True,
    )
    out = await adapter.cancel(request_id=submitted["request_id"])
    assert out["cancel_outcome"] == "moot"
    assert out["terminal"] is True and out["cancel_requested"] is False
    assert out["raven_state"] == "submitted"


async def test_duplicate_submit_does_not_launch_again_when_request_changes(tmp_path: Path) -> None:
    """A changed payload is a conflict, not a silent reuse.

    Reusing would run different work under the request_id the caller is already
    tracking, and relaunching would fork the run. Both are refused, and the
    stored fingerprint comes back so the caller can see what it disagrees with.
    """
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    first = await submit_once(adapter)
    changed = await adapter.submit(GOAL, STEP, {**REQUEST, "acceptance_criteria": ["different"]})
    assert first["request_id"] == changed["request_id"]
    assert changed["ok"] is False
    assert changed["outcome"] == "request_changed"
    assert changed["conflict"] is True
    assert changed["request_changed"] is True
    assert changed["stored_fingerprint"] != changed["submitted_fingerprint"]
    assert len(client.submits) == 1


async def test_restart_reuses_mapping_and_queries_ao_without_resubmission(tmp_path: Path) -> None:
    client = FakeClient()
    first = make_adapter(tmp_path, client)
    submitted = await submit_once(first)
    second = make_adapter(tmp_path, client)
    reconnect = await second.submit(GOAL, STEP, REQUEST)
    assert reconnect["request_id"] == submitted["request_id"]
    assert reconnect["outcome"] == "reused"
    assert len(client.submits) == 1

    status = await second.status(request_id=reconnect["request_id"])
    assert status["ok"] is True
    assert status["run_id"] == "run-1"
    assert client.statuses == [reconnect["request_id"]]
    mapped = second.store.get(reconnect["request_id"])
    assert mapped is not None and mapped.run_id == "run-1"


async def test_reserved_mapping_recovers_existing_ao_request(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    recovered = await adapter.submit(GOAL, STEP, REQUEST)
    assert recovered["outcome"] == "recovered"
    assert recovered["execution_state"] == "running"
    assert recovered["run_id"] == "run-1"
    assert client.statuses == [request_id]
    assert client.submits == []
    mapped = adapter.store.get(request_id)
    assert mapped is not None and mapped.run_id == "run-1"


async def test_reserved_mapping_reports_stale_when_ao_confirms_missing(
    tmp_path: Path,
) -> None:
    """AO confirming the request is missing must never trigger a resubmit."""
    client = FakeClient()
    client.status_exit = 2
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    outcome = await adapter.submit(GOAL, STEP, REQUEST)
    assert outcome["outcome"] == "stale_mapping"
    assert outcome["run_not_found"] is True
    assert client.statuses == [request_id]
    # An existing mapping is never auto-resubmitted, on any path.
    assert client.submits == []
    # The condition is durable, so a restart keeps reporting it.
    stored = adapter.store.get(request_id)
    assert stored is not None and stored.state == "stale_mapping"
    again = await make_adapter(tmp_path, client).submit(GOAL, STEP, REQUEST)
    assert again["outcome"] == "stale_mapping"
    assert client.submits == []


async def test_reserved_mapping_status_failure_keeps_provisional_slot(tmp_path: Path) -> None:
    client = FakeClient()
    client.status_error = CliInvocationError("status timed out")
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    outcome = await adapter.submit(GOAL, STEP, REQUEST)
    assert outcome["outcome"] == "cli_failed"
    assert outcome["command"] == "code-run-status"
    assert outcome["capacity_provisional"] is True
    assert adapter.capacity.holds(request_id) is True
    assert client.submits == []


async def test_reserved_mapping_does_not_launch_changed_request(tmp_path: Path) -> None:
    client = FakeClient()
    client.status_exit = 2
    adapter = make_adapter(tmp_path, client)
    await submit_once(adapter)
    request_id = derive_request_id(GOAL, STEP)
    adapter.store.update(request_id, state="reserved", submitted_at=None)
    changed = await adapter.submit(GOAL, STEP, {**REQUEST, "task": "changed"})
    assert changed["outcome"] == "request_changed"
    assert len(client.submits) == 1


async def test_result_for_different_request_id_is_not_consumed(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    client.result_response = result_payload("some-other-request")
    out = await adapter.result(request_id=submitted["request_id"])
    assert out["ok"] is False
    assert out["outcome"] == "malformed_response"
    assert "accepted_commit" not in out


def test_result_validator_rejects_nonformal_or_unknown_contract_values() -> None:
    with pytest.raises(MalformedResponseError):
        validate_result_shape({"stdout": "accepted"})
    invalid = result_payload(derive_request_id(GOAL, STEP), change_acceptance="blocked")
    with pytest.raises(MalformedResponseError):
        validate_result_shape(invalid)


def test_corrupt_mapping_is_not_treated_as_absent(tmp_path: Path) -> None:
    store = MappingStore(tmp_path / "mapping")
    request_id = derive_request_id(GOAL, STEP)
    store.root.mkdir(parents=True)
    (store.root / f"{request_id}.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(MappingCorruptError):
        store.reserve(
            RunMapping(
                request_id=request_id,
                parent_goal_id=GOAL,
                parent_step_id=STEP,
                request_fingerprint="fingerprint",
            )
        )


class _FakeProcess:
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"{}", b""

    async def wait(self) -> int:
        return 0


def test_client_builds_exact_ao_subcommands_and_request_file(tmp_path: Path) -> None:
    client = AoCliClient(
        ["python3", "-m", "orchestrator.cli"],
        runs_dir=tmp_path / "runs",
        working_dir=tmp_path / "ao",
    )
    assert client.build_argv(client.executable, COMMAND_STATUS, "req-1") == [
        "python3", "-m", "orchestrator.cli", "code-run-status", "req-1"
    ]
    assert client.build_argv(client.executable, COMMAND_RESULT, "req-1")[3] == "code-run-result"
    assert client.build_argv(client.executable, COMMAND_CANCEL, "req-1")[3] == "code-run-cancel"
    assert client.build_argv(client.executable, COMMAND_SUBMIT)[3] == "code-run-submit"

    observed: dict[str, Any] = {}

    async def fake_create(*argv: str, **kwargs: Any) -> _FakeProcess:
        observed["argv"] = list(argv)
        observed["stdin"] = kwargs["stdin"]
        observed["cwd"] = kwargs["cwd"]
        request_path = Path(argv[argv.index("--request") + 1])
        observed["request"] = json.loads(request_path.read_text())
        return _FakeProcess()

    async def call() -> CliResult:
        return await client.submit({"request_id": "req-1", "secret_prompt": "private"})

    with patch("raven_ao.client.asyncio.create_subprocess_exec", side_effect=fake_create):
        result = asyncio.run(call())
    argv = observed["argv"]
    assert argv[:4] == ["python3", "-m", "orchestrator.cli", "code-run-submit"]
    assert observed["request"] == {
        "request_id": "req-1", "secret_prompt": "private"
    }
    assert "private" not in argv
    assert argv[4:6] == ["--runs-dir", str(tmp_path / "runs")]
    assert argv[-2] == "--request"
    assert observed["stdin"] is asyncio.subprocess.DEVNULL
    assert observed["cwd"] == str(tmp_path / "ao")
    assert result.exit_code == 0


def test_each_manifest_factory_returns_one_raven_tool(tmp_path: Path) -> None:
    ctx = {
        "executable": ["python3", "-m", "orchestrator.cli"],
        "runs_dir": str(tmp_path / "runs"),
        "mapping_dir": str(tmp_path / "mapping"),
    }
    tools = [
        make_submit_tool(ctx),
        make_status_tool(ctx),
        make_result_tool(ctx),
        make_cancel_tool(ctx),
        make_events_tool(ctx),
    ]
    assert [type(tool) for tool in tools] == [
        SubmitTool, StatusTool, ResultTool, CancelTool, EventsTool
    ]


def test_manifest_has_five_single_tool_contributions() -> None:
    manifest_path = Path(__file__).parents[1] / "raven_ao" / "raven-plugin.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    contributions = manifest["plugin"]["contributes"]["tools"]
    assert [item["name"] for item in contributions] == [
        "ao_code_run_submit",
        "ao_code_run_status",
        "ao_code_run_result",
        "ao_code_run_cancel",
        "ao_code_run_events",
    ]
    assert all(item["factory"].startswith("raven_ao.tools:") for item in contributions)
    assert "working_dir" in manifest["plugin"]["config_schema"]
    assert manifest["plugin"]["config_schema"]["max_concurrent_code_runs"] == {
        "type": "integer"
    }


# --- AO event pages, the durable cursor, and recovery across restart --------


async def test_events_resume_from_cursor_and_survive_a_restart(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    request_id = submitted["request_id"]

    client.event_page = event_page(request_id, [event(0), event(1), event(2)])
    first = await adapter.events(request_id=request_id)
    assert [e["sequence"] for e in first["events"]] == [0, 1, 2]
    # AO numbers its log from 0, so a fresh reader must ask with --after-seq -1.
    assert client.event_reads == [(request_id, -1)]
    assert first["last_seen_event_seq"] == 2

    # A brand new adapter over the same durable directory is a restart: it must
    # ask AO for sequences strictly greater than the stored cursor, not re-read.
    restarted = make_adapter(tmp_path, client)
    client.event_page = event_page(request_id, [event(3)])
    second = await restarted.events(request_id=request_id)
    assert client.event_reads[-1] == (request_id, 2)
    assert [e["sequence"] for e in second["events"]] == [3]
    assert second["last_seen_event_seq"] == 3

    client.event_page = event_page(request_id, [])
    await restarted.events(request_id=request_id)
    assert client.event_reads[-1] == (request_id, 3)


async def test_events_persist_the_cursor_with_the_rest_of_the_mapping(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    submitted = await submit_once(adapter)
    request_id = submitted["request_id"]
    client.event_page = event_page(request_id, [event(4, "task_completed")])

    out = await adapter.events(request_id=request_id)
    assert out["ok"] is True and out["outcome"] == "ok"
    assert out["event_types"] == ["task_completed"]

    mapping = MappingStore(tmp_path / "mapping").get(request_id)
    assert mapping is not None
    assert mapping.last_seen_event_seq == 4
    # Identity survives the restart that reads events.
    assert mapping.request_id == request_id
    assert mapping.parent_goal_id == GOAL and mapping.parent_step_id == STEP


async def test_events_refuse_a_cursor_behind_the_stored_position(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    client.event_page = event_page(request_id, [event(5)])
    await adapter.events(request_id=request_id)

    behind = await adapter.events(request_id=request_id, after_seq=1)
    assert behind["ok"] is False and behind["outcome"] == "invalid_request"
    # The stale range was never even requested, so nothing was re-read.
    assert client.event_reads == [(request_id, -1)]


async def test_events_never_advance_the_cursor_past_an_invalid_page(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]

    for bad in (
        event_page(request_id, [event(1), event(1)]),                      # not monotonic
        event_page(request_id, [event(1, timestamp="yesterday")]),        # not ISO-8601
        event_page(request_id, [event(1, event_type="")]),                # no type
        event_page("other-request", [event(1)]),                          # wrong request
        event_page(request_id, [{"sequence": 1.5, "event_type": "x", "timestamp": "2026-10-02T00:00:00+00:00"}]),
    ):
        client.event_page = bad
        out = await adapter.events(request_id=request_id)
        assert out["ok"] is False, bad
        assert out["outcome"] == "malformed_response", bad
        stored = MappingStore(tmp_path / "mapping").get(request_id)
        assert stored is not None and stored.last_seen_event_seq == -1, bad


async def test_events_never_derive_a_verdict_or_release_capacity(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    # AO's event names whatever it likes; a page full of terminal-sounding
    # words is still not a verdict and must not free the outer-AO slot.
    client.event_page = event_page(
        request_id,
        [event(1, "run_accepted"), event(2, "run_completed")],
    )
    out = await adapter.events(request_id=request_id)
    assert "change_acceptance" not in out
    assert adapter.capacity.holds(request_id) is True


async def test_repeated_submit_resolves_to_the_same_coderun(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    first = await submit_once(adapter)
    again = await submit_once(adapter)
    assert first["request_id"] == again["request_id"]
    assert again["launched"] is False
    assert len(client.submits) == 1


async def test_duplicate_submit_with_a_changed_payload_is_refused(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    await submit_once(adapter)
    conflict = await adapter.submit(GOAL, STEP, {**REQUEST, "package_id": "different"})
    assert conflict["ok"] is False
    assert conflict["outcome"] == "request_changed"
    assert conflict["conflict"] is True
    assert conflict["stored_fingerprint"] != conflict["submitted_fingerprint"]
    # Neither silently reused nor silently relaunched.
    assert len(client.submits) == 1
    assert "accepted_commit" not in conflict and "change_acceptance" not in conflict


async def test_missing_ao_request_reports_stale_mapping_without_resubmitting(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    client.status_exit = 2

    out = await adapter.status(request_id=request_id)
    assert out["ok"] is False
    assert out["outcome"] == "stale_mapping"
    assert out["run_not_found"] is True

    resubmitted = await submit_once(adapter)
    assert resubmitted["ok"] is False
    assert resubmitted["outcome"] == "stale_mapping"
    assert len(client.submits) == 1

    # AO confirmed the run is not there, so its outer-AO slot is released.
    assert adapter.capacity.holds(request_id) is False
    stored = MappingStore(tmp_path / "mapping").get(request_id)
    assert stored is not None and stored.state == "stale_mapping"


async def test_cancel_requested_survives_restart_and_never_resubmits(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    cancelled = await adapter.cancel(request_id=request_id, reason="no longer needed")
    assert cancelled["cancel_outcome"] == "cancel_requested"
    assert cancelled["cancel_requested"] is True

    # Restart: a new adapter, the same durable directory.
    restarted = make_adapter(tmp_path, client)
    resubmit = await restarted.submit(GOAL, STEP, REQUEST)
    assert resubmit["outcome"] == "cancel_pending"
    assert len(client.submits) == 1

    mapping = MappingStore(tmp_path / "mapping").get(request_id)
    assert mapping is not None
    assert mapping.cancel_requested is True
    assert mapping.state == "cancel_requested"

    # AO's terminal cancelled result is what finally settles it.
    client.result_response = result_payload(
        request_id, execution_status="cancelled", change_acceptance="pending"
    )
    settled = await restarted.result(request_id=request_id)
    assert settled["cancel_confirmed"] is True
    assert settled["execution_status"] == "cancelled"
    assert restarted.capacity.holds(request_id) is False


async def test_interrupted_stays_its_own_recoverable_state(tmp_path: Path) -> None:
    """AO reporting ``interrupted`` is observable, not a malformed response.

    It is not terminal, not a failure, and not a cancellation, so the run keeps
    its outer-AO slot and stays resumable.
    """
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    client.status_response = {
        "request_id": request_id,
        "execution_state": "interrupted",
        "change_acceptance": "not_evaluated",
    }

    out = await adapter.status(request_id=request_id)
    assert out["ok"] is True
    assert out["outcome"] == "interrupted"
    assert out["lifecycle"] == "interrupted"
    assert out["interrupted"] is True
    assert out["recoverable"] is True
    assert out["terminal"] is False
    # Neither an acceptance value nor a released slot is invented.
    assert "change_acceptance" not in out
    assert adapter.capacity.holds(request_id) is True


async def test_pending_cancel_keeps_its_slot_and_stays_nonterminal(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    client.status_response = status_payload(
        request_id,
        execution_state="cancel_requested",
        execution_status="running",
    )
    live = await adapter.status(request_id=request_id)
    assert live["lifecycle"] == "cancel_requested"
    assert live["terminal"] is False
    # A pending cancellation still holds the outer-AO slot.
    assert adapter.capacity.holds(request_id) is True


async def test_outer_ao_concurrency_limit_is_enforced_across_processes(
    tmp_path: Path,
) -> None:
    first = FakeClient()
    adapter = make_adapter(tmp_path, first)
    assert adapter.max_concurrent_code_runs == 1

    launched = await submit_once(adapter)
    assert launched["ok"] is True
    assert adapter.capacity.active_count() == 1

    # A second Raven process over the same durable directory is refused the slot.
    other = make_adapter(tmp_path, FakeClient())
    blocked = await other.submit("goal-8", "step-1", REQUEST)
    assert blocked["ok"] is False
    assert blocked["outcome"] == "capacity_exhausted"
    assert blocked["limit"] == 1
    assert other.client.submits == []

    # Only an authoritative terminal answer frees it.
    first.status_response = status_payload(
        launched["request_id"], execution_state="terminal", execution_status="completed"
    )
    settled = await adapter.status(request_id=launched["request_id"])
    assert settled["terminal"] is True
    assert other.capacity.active_count() == 0

    assert (await other.submit("goal-8", "step-1", REQUEST))["ok"] is True


async def test_events_cli_invocation_failure_keeps_the_cursor(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    client.event_page = event_page(request_id, [event(7)])
    await adapter.events(request_id=request_id)

    client.events_error = CliInvocationError("AO events call timed out")
    failed = await adapter.events(request_id=request_id)
    assert failed["ok"] is False
    assert failed["outcome"] == "cli_failed"
    # Infrastructure failure is not a verdict and does not lose the cursor.
    assert failed["events"] == []
    stored = MappingStore(tmp_path / "mapping").get(request_id)
    assert stored is not None and stored.last_seen_event_seq == 7


def test_events_client_builds_the_after_seq_argv(tmp_path: Path) -> None:
    client = AoCliClient(
        ["python3", "-m", "orchestrator.cli"],
        runs_dir=tmp_path / "runs",
        working_dir=tmp_path / "ao",
    )
    argv = client.build_argv(client.executable, COMMAND_EVENTS, "req-1")
    assert argv[3] == "code-run-events" and argv[4] == "req-1"
    assert COMMAND_EVENTS in AO_COMMANDS

    observed: dict[str, Any] = {}

    class _Proc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"events": []}', b""

        async def wait(self) -> int:
            return 0

    async def fake_create(*argv: str, **kwargs: Any) -> _Proc:
        observed["argv"] = list(argv)
        return _Proc()

    async def call() -> CliResult:
        return await client.events("req-1", 12)

    with patch("raven_ao.client.asyncio.create_subprocess_exec", side_effect=fake_create):
        asyncio.run(call())
    tail = observed["argv"]
    assert tail[:5] == ["python3", "-m", "orchestrator.cli", "code-run-events", "req-1"]
    assert tail[-2:] == ["--after-seq", "12"]


def test_max_concurrent_code_runs_defaults_to_one_and_rejects_nonsense(
    tmp_path: Path,
) -> None:
    from raven_ao.config import ConfigError, resolve_config

    base = {
        "executable": ["python3", "-m", "orchestrator.cli"],
        "runs_dir": str(tmp_path / "runs"),
        "mapping_dir": str(tmp_path / "mapping"),
    }
    assert resolve_config(base).max_concurrent_code_runs == 1
    assert resolve_config({**base, "max_concurrent_code_runs": 3}).max_concurrent_code_runs == 3
    for bad in (0, -1, True, "2", 1.5):
        with pytest.raises(ConfigError):
            resolve_config({**base, "max_concurrent_code_runs": bad})


async def test_event_page_publishes_a_first_run_id_and_the_cursor_together(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    # The submit payload carried no run directory, so Raven knows no run_id yet.
    assert MappingStore(tmp_path / "mapping").get(request_id).run_id is None

    client.event_page = event_page(
        request_id, [event(0, "run_started", run_id="run-9")], run_id="run-9"
    )
    consumed = await adapter.events(request_id=request_id)
    assert consumed["ok"] is True
    assert consumed["run_id"] == "run-9"

    stored = MappingStore(tmp_path / "mapping").get(request_id)
    # The run identity and the event position are one durable fact: a restart
    # must never see one without the other.
    assert stored is not None
    assert stored.run_id == "run-9"
    assert stored.last_seen_event_seq == 0


async def test_event_page_rejects_a_run_id_the_mapping_never_knew(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    adapter.store.update(request_id, run_id="run-1")

    client.event_page = event_page(request_id, [event(1)], run_id="run-other")
    rejected = await adapter.events(request_id=request_id)
    assert rejected["ok"] is False
    assert rejected["outcome"] == "malformed_response"
    assert "run-other" in rejected["error"]
    assert rejected["events"] == []
    assert MappingStore(tmp_path / "mapping").get(request_id).last_seen_event_seq == -1


async def test_event_page_rejects_an_event_belonging_to_another_run(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    adapter.store.update(request_id, run_id="run-1")

    # The page claims run-1 but carries run-2's event: a mixed response that
    # would attribute one run's progress to another run's cursor.
    client.event_page = event_page(request_id, [event(1, run_id="run-2")], run_id="run-1")
    rejected = await adapter.events(request_id=request_id)
    assert rejected["ok"] is False
    assert rejected["outcome"] == "malformed_response"
    assert "run-2" in rejected["error"]
    assert MappingStore(tmp_path / "mapping").get(request_id).last_seen_event_seq == -1


async def test_event_page_rejects_an_event_without_a_run_id(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    adapter.store.update(request_id, run_id="run-1")

    # AO stamps every published event with its run. A page that omits it (or
    # sends an empty or non-string value) is a shape Raven cannot attribute to
    # a run, so it is refused and the durable cursor must not move.
    without = {k: v for k, v in event(1).items() if k != "run_id"}
    for bad_event in (without, {**event(1), "run_id": ""}, {**event(1), "run_id": 7}):
        client.event_page = event_page(request_id, [bad_event], run_id="run-1")
        rejected = await adapter.events(request_id=request_id)
        assert rejected["ok"] is False
        assert rejected["outcome"] == "malformed_response"
        assert rejected["events"] == []
        stored = MappingStore(tmp_path / "mapping").get(request_id)
        assert stored is not None and stored.last_seen_event_seq == -1


async def test_empty_event_page_is_valid_before_ao_publishes_a_run_id(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    assert MappingStore(tmp_path / "mapping").get(request_id).run_id is None

    # Nothing published yet: no run identity exists, and an empty page is not a
    # claim about one, so it still reads and still asks AO with the cursor.
    client.event_page = event_page(request_id, [])
    out = await adapter.events(request_id=request_id)
    assert out["ok"] is True
    assert out["events"] == []
    assert out["last_seen_event_seq"] == -1
    assert client.event_reads == [(request_id, -1)]


async def test_matching_ao_event_page_shape_is_consumed(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = (await submit_once(adapter))["request_id"]
    adapter.store.update(request_id, run_id="run-1")

    # Exactly AO's shape: request_id, run_id, and per event a sequence, an
    # event_type and an ISO-8601 timestamp.
    client.event_page = {
        "request_id": request_id,
        "run_id": "run-1",
        "events": [
            {
                "request_id": request_id,
                "run_id": "run-1",
                "sequence": 0,
                "event_type": "run_started",
                "timestamp": "2026-10-02T00:00:00+00:00",
            },
            {
                "request_id": request_id,
                "run_id": "run-1",
                "sequence": 1,
                "event_type": "task_completed",
                "timestamp": "2026-10-02T00:00:05+00:00",
            },
        ],
    }
    consumed = await adapter.events(request_id=request_id)
    assert consumed["ok"] is True
    assert consumed["event_count"] == 2
    assert consumed["last_seen_event_seq"] == 1
    assert consumed["event_types"] == ["run_started", "task_completed"]
    assert [e["sequence"] for e in consumed["events"]] == [0, 1]
    # An event page is a projection, never a verdict: nothing is released.
    assert adapter.capacity.active_count() == 1


class _BlockingSubmitClient(FakeClient):
    """An AO boundary whose first submit blocks until the test releases it.

    Used to hold one Raven process inside its submit call, so a second process
    provably reaches its own submit *while the first is still in flight* rather
    than after it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, request: dict[str, Any]) -> CliResult:
        if not self.submits:
            self.entered.set()
            await self.release.wait()
        return await super().submit(request)


async def test_concurrent_same_identity_submits_once_and_keeps_one_slot(tmp_path: Path) -> None:
    first_client = _BlockingSubmitClient()
    second_client = FakeClient()
    first = make_adapter(tmp_path, first_client)
    second = make_adapter(tmp_path, second_client)

    # The first process is inside AO's submit when the second one starts.
    inflight = asyncio.create_task(submit_once(first))
    await asyncio.wait_for(first_client.entered.wait(), timeout=5)

    # A capacity failure would be the wrong outcome here: this is a duplicate of
    # the same identity, not a second concurrent run.
    follower = asyncio.create_task(submit_once(second))
    await asyncio.sleep(0.05)
    assert not follower.done(), "the second process raced past the first one's submit"

    first_client.release.set()
    launched = await asyncio.wait_for(inflight, timeout=5)
    followed = await asyncio.wait_for(follower, timeout=5)

    # Exactly one AO submit for one identity, whoever won the race.
    assert len(first_client.submits) + len(second_client.submits) == 1
    assert launched["request_id"] == followed["request_id"] == derive_request_id(GOAL, STEP)

    # The follower must not have probed status into a premature stale verdict
    # while the first submit was still unknown to AO.
    assert followed["ok"] is True
    assert followed["outcome"] != "stale_mapping"
    assert "run_not_found" not in followed

    # One run, one outer-AO slot -- the follower never stole the first call's.
    assert first.capacity.active_count() == 1
    assert second.capacity.active_count() == 1
    assert first.capacity.active_request_ids() == [derive_request_id(GOAL, STEP)]

    stored = MappingStore(tmp_path / "mapping").get(launched["request_id"])
    assert stored is not None
    assert stored.state == "submitted"


async def test_reserved_recovery_releases_capacity_on_terminal_status(tmp_path: Path) -> None:
    """A recovered reservation AO has already settled must give its slot back."""
    client = FakeClient()
    request_id = derive_request_id(GOAL, STEP)
    client.status_response = status_payload(
        request_id, execution_state="terminal", execution_status="completed"
    )
    adapter = make_adapter(tmp_path, client)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    assert adapter.capacity.acquire(request_id, 1)[0]
    assert adapter.capacity.holds(request_id) is True

    recovered = await adapter.submit(GOAL, STEP, REQUEST)
    assert recovered["outcome"] == "recovered"
    assert recovered["execution_state"] == "terminal"
    assert recovered["capacity_released"] is True
    assert client.submits == []
    assert adapter.capacity.holds(request_id) is False
    assert adapter.capacity.active_count() == 0


async def test_phase2_mapping_without_marker_adopts_exactly_one_slot(tmp_path: Path) -> None:
    """A pre-guard mapping reconnects to its live run and claims one slot."""
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    # The shape a Phase 2 process left behind: submitted, but with no capacity
    # marker, because the outer guard did not exist yet.
    adapter.store.update(
        request_id, state="submitted", submitted_at="2026-10-02T00:00:00+00:00", run_id="run-1"
    )
    assert adapter.capacity.holds(request_id) is False

    resumed = await make_adapter(tmp_path, client).submit(GOAL, STEP, REQUEST)
    assert resumed["outcome"] == "reused"
    assert resumed["request_id"] == request_id
    assert resumed["capacity_adopted"] is True
    assert client.submits == []
    assert make_adapter(tmp_path, client).capacity.active_request_ids() == [request_id]

    # A second Raven process reuses the adopted slot instead of taking another.
    again = await make_adapter(tmp_path, client).submit(GOAL, STEP, REQUEST)
    assert again["outcome"] == "reused"
    assert client.submits == []
    assert make_adapter(tmp_path, client).capacity.active_count() == 1


async def test_phase2_reserved_mapping_claims_slot_before_status_probe(tmp_path: Path) -> None:
    """An uncertain Phase 2 reservation is counted while AO is reconciled."""
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )

    recovered = await adapter.submit(GOAL, STEP, REQUEST)
    assert recovered["outcome"] == "recovered"
    assert recovered["execution_state"] == "running"
    assert recovered["capacity_provisional"] is True
    assert client.statuses == [request_id]
    assert client.submits == []
    assert adapter.capacity.active_request_ids() == [request_id]
    assert adapter.store.get(request_id).state == "submitted"


async def test_phase2_terminal_mapping_reconciles_without_marker(tmp_path: Path) -> None:
    """A terminal pre-guard mapping releases its provisional slot and returns AO's verdict."""
    client = FakeClient()
    request_id = derive_request_id(GOAL, STEP)
    client.status_response = status_payload(
        request_id, execution_state="terminal", execution_status="completed",
        change_acceptance="accepted", result_available=True,
    )
    adapter = make_adapter(tmp_path, client)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
            state="submitted",
            submitted_at="2026-10-02T00:00:00+00:00",
            run_id="run-1",
        )
    )

    reconciled = await adapter.submit(GOAL, STEP, REQUEST)
    assert reconciled["outcome"] == "reconciled"
    assert reconciled["run_id"] == "run-1"
    assert reconciled["execution_state"] == "terminal"
    assert reconciled["change_acceptance"] == "accepted"
    assert reconciled["capacity_released"] is True
    assert adapter.capacity.active_count() == 0
    assert client.submits == []


async def test_phase2_status_failure_keeps_provisional_slot(tmp_path: Path) -> None:
    """A failed status probe stays structured and cannot free uncertain capacity."""
    client = FakeClient()
    client.status_error = CliInvocationError("temporary AO outage")
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )

    failed = await adapter.submit(GOAL, STEP, REQUEST)
    assert failed["ok"] is False
    assert failed["outcome"] == "cli_failed"
    assert failed["command"] == "code-run-status"
    assert failed["capacity_provisional"] is True
    assert adapter.capacity.holds(request_id) is True
    assert client.submits == []


async def test_resumed_run_without_marker_keeps_identity_when_cap_is_full(tmp_path: Path) -> None:
    """A full cap must neither launch a run nor lose the request's identity."""
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    request_id = derive_request_id(GOAL, STEP)
    effective = {**REQUEST, "request_id": request_id, "parent_goal_id": GOAL, "parent_step_id": STEP}
    adapter.store.reserve(
        RunMapping(
            request_id=request_id,
            parent_goal_id=GOAL,
            parent_step_id=STEP,
            request_fingerprint=fingerprint_request(effective),
        )
    )
    adapter.store.update(
        request_id, state="submitted", submitted_at="2026-10-02T00:00:00+00:00", run_id="run-1"
    )
    assert adapter.capacity.acquire("some-other-live-run", 1)[0]

    resumed = await make_adapter(tmp_path, client).submit(GOAL, STEP, REQUEST)
    assert resumed["outcome"] == "reused"
    assert resumed["request_id"] == request_id
    assert resumed["capacity_adopted"] is False
    assert client.submits == []
    assert client.statuses == []
    assert make_adapter(tmp_path, client).capacity.active_request_ids() == ["some-other-live-run"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
