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
    COMMAND_CANCEL,
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
    ResultTool,
    StatusTool,
    SubmitTool,
    make_cancel_tool,
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
        self.status_response: dict[str, Any] | None = None
        self.result_response: dict[str, Any] | None = None
        self.cancel_response: dict[str, Any] | None = None
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
    client = FakeClient()
    adapter = make_adapter(tmp_path, client)
    first = await submit_once(adapter)
    changed = await adapter.submit(GOAL, STEP, {**REQUEST, "acceptance_criteria": ["different"]})
    assert first["request_id"] == changed["request_id"]
    assert changed["outcome"] == "reused"
    assert changed["request_changed"] is True
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


async def test_reserved_mapping_retries_only_after_ao_confirms_missing(tmp_path: Path) -> None:
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
    recovered = await adapter.submit(GOAL, STEP, REQUEST)
    assert recovered["outcome"] == "recovered"
    assert client.statuses == [request_id]
    assert len(client.submits) == 1


async def test_reserved_mapping_reports_status_invocation_failure_as_unconfirmed(tmp_path: Path) -> None:
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
    assert outcome["outcome"] == "launch_unconfirmed"
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
    ]
    assert [type(tool) for tool in tools] == [SubmitTool, StatusTool, ResultTool, CancelTool]


def test_manifest_has_four_single_tool_contributions() -> None:
    manifest_path = Path(__file__).parents[1] / "raven_ao" / "raven-plugin.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    contributions = manifest["plugin"]["contributes"]["tools"]
    assert [item["name"] for item in contributions] == [
        "ao_code_run_submit",
        "ao_code_run_status",
        "ao_code_run_result",
        "ao_code_run_cancel",
    ]
    assert all(item["factory"].startswith("raven_ao.tools:") for item in contributions)
    assert "working_dir" in manifest["plugin"]["config_schema"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
