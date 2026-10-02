"""AO's own vocabularies and wire shapes, as the adapter must read them.

Every value agent-orchestrator (AO) can report is *data to be validated*, never
prose to be interpreted. This module owns three things:

* the closed value sets AO uses, transcribed from AO's own source rather than
  guessed -- they are split by axis and by surface, because AO reports
  different sets for a durable status record and for a formal result;
* the shape a response must have before the adapter may read meaning out of it;
* the rule that decides whether an accepted artifact may be surfaced.

The sets here are pinned to what AO actually emits, and the tests assert them
against AO's real output. In particular AO's two axes are kept apart:

``execution_state``
    The *durable* lifecycle word, from ``code-run-status``. While the run is
    live it is ``queued``/``running``/``cancel_requested``; once it settles it is
    one of ``terminal``/``blocked``/``failed``/``cancelled``.

``execution_status``
    The word carried by a *formal* ``CodeRunResult``: ``pending``/``running``/
    ``completed``/``blocked``/``failed``/``cancelled``. A result never carries
    ``execution_state``.

``change_acceptance``
    Adjudication of the change, always separate from the two above: exactly
    ``pending``/``accepted``/``rejected`` on a result. A durable status record
    adds ``not_evaluated``, which is the projection for a *live* run whose
    change has not been adjudicated at all -- it distinguishes "still running,
    nothing judged" from "running and rejected". ``not_evaluated`` is never a
    value of a formal result, and ``blocked``/``failed``/``cancelled`` are never
    values of either acceptance axis: AO reports those on the execution axis,
    and folding them in here is exactly the conflation this module exists to
    prevent.

The last owned rule is the adapter's core safety property. A CodeRun can finish
its execution, exit 0, and still be ``rejected`` with
``failure_class = "scope_violation"``. Acceptance is therefore decided *only* by
the structured ``change_acceptance`` field of a formal result, and an accepted
artifact is surfaced only when that field is ``accepted`` **and** the same
response carries a concrete artifact **and** it reports
``verification_complete``. Nothing else -- not the process exit code, not a
terminal execution state, not Raven's own bookkeeping, not stdout prose -- can
promote a result to accepted.
"""

from __future__ import annotations

import json
from typing import Any, Final

# --- AO vocabularies ------------------------------------------------------
#
# Transcribed from AO's ``orchestrator.code_run`` (result vocabularies) and
# ``orchestrator.code_run_service`` (durable lifecycle vocabularies).

#: ``change_acceptance`` on a *formal* ``CodeRunResult``. Exactly three values.
RESULT_ACCEPTANCE_VALUES: Final = frozenset({"pending", "accepted", "rejected"})

#: ``change_acceptance`` on a *durable status* record: the three result values
#: plus the live-run projection AO substitutes while nothing has been judged.
STATUS_ACCEPTANCE_VALUES: Final = frozenset(
    {"pending", "accepted", "rejected", "not_evaluated"}
)

#: The live-run projection, valid on status only.
NOT_EVALUATED: Final = "not_evaluated"

#: The single acceptance value that may expose an artifact.
ACCEPTANCE_ACCEPTED: Final = "accepted"

#: Durable ``execution_state`` reported by ``code-run-status``.
LIVE_EXECUTION_STATES: Final = frozenset({"queued", "running", "cancel_requested"})
TERMINAL_EXECUTION_STATES: Final = frozenset(
    {"terminal", "blocked", "failed", "cancelled"}
)
EXECUTION_STATES: Final = LIVE_EXECUTION_STATES | TERMINAL_EXECUTION_STATES

#: ``execution_status`` carried by a formal ``CodeRunResult``.
TERMINAL_EXECUTION_STATUSES: Final = frozenset(
    {"completed", "blocked", "failed", "cancelled"}
)
EXECUTION_STATUSES: Final = frozenset({"pending", "running"}) | (
    TERMINAL_EXECUTION_STATUSES
)

#: Every ``failure_class`` AO can report, preserved verbatim on the way out.
#: ``none`` is the real "no failure, evidence still incomplete" class and is
#: present on healthy accepted runs, so it must not be treated as absent.
FAILURE_CLASSES: Final = frozenset(
    {
        "none",
        "scope_violation",
        "dependency_conflict",
        "base_drift",
        "needs_input",
        "infrastructure",
        "verification",
        "cancellation",
    }
)

_MISSING: Final = object()


class MalformedResponseError(Exception):
    """The CLI output is not a formal AO response this adapter may read.

    Raised for non-JSON output, a non-object payload, a missing required field,
    or a value outside a closed AO set. Callers must surface it as a failure and
    must never fall back to interpreting the raw text.
    """


# --- parsing --------------------------------------------------------------


def parse_response(raw: str) -> dict[str, Any]:
    """Parse CLI stdout into a JSON object, or raise :class:`MalformedResponseError`."""
    text = (raw or "").strip()
    if not text:
        raise MalformedResponseError("empty response from the AO CLI")
    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        # A human-readable line here means the CLI failed in a way we did not
        # model. It is never mined for meaning.
        raise MalformedResponseError(f"response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedResponseError("response was not a JSON object")
    return payload


def cli_error_detail(stdout: str, stderr: str) -> str:
    """Render AO's ``code-run-*`` failure envelope as readable detail.

    AO reports a refused command on **stdout** as
    ``{"error": {"type": ..., "message": ..., "problems": [...]}}`` with exit
    status 2, so stderr alone would read as an empty failure. The envelope is
    decoded when present purely to build a message for the operator; nothing is
    inferred about the run's outcome from it.
    """
    for stream in (stdout, stderr):
        text = (stream or "").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            parts = [str(error.get("type") or "error"), str(error.get("message") or "")]
            problems = error.get("problems")
            if isinstance(problems, list) and problems:
                parts.append("; ".join(str(p) for p in problems))
            return ": ".join(part for part in parts if part)
        return text
    return "no diagnostic output"


# --- field readers --------------------------------------------------------


def _require_str(payload: dict[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key, _MISSING)
    if value is _MISSING or value is None:
        raise MalformedResponseError(f"{label} is missing {key!r}")
    if not isinstance(value, str) or not value.strip():
        raise MalformedResponseError(f"{label} field {key!r} was not a non-empty string")
    return value


def _require_choice(
    payload: dict[str, Any], key: str, allowed: frozenset[str], *, label: str
) -> str:
    value = payload.get(key, _MISSING)
    if value is _MISSING or value is None:
        raise MalformedResponseError(f"{label} is missing {key!r}")
    if not isinstance(value, str) or value not in allowed:
        raise MalformedResponseError(f"{label} has unknown {key} {value!r}")
    return value


def _optional_str(payload: dict[str, Any], key: str, *, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MalformedResponseError(f"{label} field {key!r} was not a string")
    return value


def failure_detail(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Preserve AO's ``failure_class`` / ``failure_reason`` pair verbatim.

    ``failure_class`` is validated against AO's set so a typo cannot masquerade
    as a known class, but the value returned to the caller is never rewritten. A
    reason is prose and is passed through untouched. Absence is reported as
    ``None`` rather than substituted, so a response that omits the field is not
    silently read as ``none``.
    """
    raw_class = payload.get("failure_class", _MISSING)
    if raw_class is _MISSING or raw_class is None:
        return {"failure_class": None, "failure_reason": None}
    if not isinstance(raw_class, str) or raw_class not in FAILURE_CLASSES:
        raise MalformedResponseError(f"{label} has unknown failure_class {raw_class!r}")
    reason = payload.get("failure_reason")
    if reason is not None and not isinstance(reason, str):
        raise MalformedResponseError(f"{label} field 'failure_reason' was not a string")
    return {"failure_class": raw_class, "failure_reason": reason}


def _concrete_artifact(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the accepted artifact only when the response carries a real one.

    AO types ``accepted_commit`` and ``accepted_patch`` as strings -- a rejected
    run carries both as ``""`` rather than omitting them. Empty is therefore
    "no artifact", not "an empty artifact".
    """
    commit = payload.get("accepted_commit")
    patch = payload.get("accepted_patch")
    commit_ok = isinstance(commit, str) and commit.strip() != ""
    patch_ok = isinstance(patch, str) and patch.strip() != ""
    if not commit_ok and not patch_ok:
        return None
    artifact: dict[str, Any] = {}
    if commit_ok:
        artifact["accepted_commit"] = commit
    if patch_ok:
        artifact["accepted_patch"] = patch
    return artifact


def accepted_artifact(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Decide whether accepted artifacts may be exposed, per AO's own verdict.

    All three conditions must hold:

    1. ``change_acceptance`` is exactly ``accepted``;
    2. execution completed, no failure was recorded, and AO reports complete,
       passing scope, review, and acceptance gates;
    3. the formal result carries a concrete ``accepted_commit`` or
       ``accepted_patch``;
    4. ``verification_complete`` is exactly ``true``.

    Any other combination -- including a run whose ``execution_status`` is
    ``completed``, which exited 0, or which said "done" in prose -- yields
    ``None``, and the caller then reports no accepted artifact at all.
    """
    if payload.get("change_acceptance") != ACCEPTANCE_ACCEPTED:
        return None
    if payload.get("execution_status") != "completed":
        return None
    if payload.get("failure_class") != "none":
        return None
    for gate in ("scope_evidence", "review_evidence", "acceptance_evidence"):
        evidence = payload.get(gate)
        if not isinstance(evidence, dict):
            return None
        if evidence.get("status") != "passed" or evidence.get("complete") is not True:
            return None
    artifact = _concrete_artifact(payload)
    if artifact is None:
        return None
    if payload.get("verification_complete") is not True:
        return None
    return artifact


def lifecycle(
    execution_state: str | None,
    execution_status: str | None,
    change_acceptance: str | None,
) -> str:
    """Derive one Raven-facing lifecycle word from AO's real fields only.

    This is a *projection*, never a new verdict: every branch reads an
    execution value AO itself reported, and nothing here can promote a run to
    accepted. The acceptance axis wins when AO adjudicated a change, because a
    completed execution that was rejected is ``rejected``, not ``completed``.
    """
    if change_acceptance == ACCEPTANCE_ACCEPTED:
        return "accepted"
    if change_acceptance == "rejected":
        return "rejected"
    if execution_state in LIVE_EXECUTION_STATES:
        return "cancel_requested" if execution_state == "cancel_requested" else "in_flight"
    if execution_status in TERMINAL_EXECUTION_STATUSES:
        return execution_status
    if execution_state in TERMINAL_EXECUTION_STATES:
        return execution_state
    return "unknown"


# --- surface validation ---------------------------------------------------


def validate_status_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Gate a payload as AO's *durable status* record.

    Requires the ``request_id`` it is keyed by, an ``execution_state`` from
    AO's durable vocabulary, and a ``change_acceptance`` from the status set
    (which, unlike a result, may be ``not_evaluated`` for a live run).
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError("status was not a JSON object")
    request_id = _require_str(payload, "request_id", label="status")
    execution_state = _require_choice(
        payload, "execution_state", EXECUTION_STATES, label="status"
    )
    change_acceptance = _require_choice(
        payload, "change_acceptance", STATUS_ACCEPTANCE_VALUES, label="status"
    )
    result_available = payload.get("result_available")
    if result_available is not None and not isinstance(result_available, bool):
        raise MalformedResponseError("status field 'result_available' was not a boolean")
    return {
        "request_id": request_id,
        "execution_state": execution_state,
        "execution_status": _optional_str(
            payload, "execution_status", label="status"
        ),
        "change_acceptance": change_acceptance,
        "result_available": bool(result_available),
        "ao_run_dir": _optional_str(payload, "ao_run_dir", label="status"),
        "next_action": _optional_str(payload, "next_action", label="status"),
        **failure_detail(payload, label="status"),
    }


def validate_result_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Gate a payload as a *formal* ``CodeRunResult`` before reading meaning.

    A formal result carries ``schema_version``, the ``request_id`` it answers
    for, the AO ``run_id``, an ``execution_status``, a ``change_acceptance``
    from the three-value result set, and its acceptance evidence. A response
    missing any of that is not a result; it is a CLI error, and it is reported
    as such rather than read as "probably fine".

    ``execution_state`` is deliberately *not* read here: AO does not put it on a
    result, and treating a missing field as a state would let a caller believe
    it knows a durable lifecycle word it was never given.
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError("result was not a JSON object")
    schema_version = _require_choice(
        payload, "schema_version", frozenset({"1.0"}), label="result"
    )
    request_id = _require_str(payload, "request_id", label="result")
    run_id = _require_str(payload, "run_id", label="result")
    execution_status = _require_choice(
        payload, "execution_status", EXECUTION_STATUSES, label="result"
    )
    change_acceptance = _require_choice(
        payload, "change_acceptance", RESULT_ACCEPTANCE_VALUES, label="result"
    )
    verification_complete = payload.get("verification_complete")
    if not isinstance(verification_complete, bool):
        raise MalformedResponseError("result field 'verification_complete' was not a boolean")
    for key in (
        "base_commit",
        "effective_base_commit",
        "parent_goal_id",
        "parent_step_id",
        "package_id",
        "accepted_commit",
        "accepted_patch",
        "started_at",
        "finished_at",
    ):
        if not isinstance(payload.get(key), str):
            raise MalformedResponseError(f"result field {key!r} was not a string")
    for key in ("task_evidence", "attempt_evidence", "tests"):
        if not isinstance(payload.get(key), list):
            raise MalformedResponseError(f"result field {key!r} was not a list")
    for key in ("scope_evidence", "review_evidence", "acceptance_evidence"):
        if not isinstance(payload.get(key), dict):
            raise MalformedResponseError(f"result field {key!r} was not an object")
    if "failure_class" not in payload or not isinstance(payload["failure_class"], str):
        raise MalformedResponseError("result is missing string field 'failure_class'")
    if not isinstance(payload.get("failure_reason"), str):
        raise MalformedResponseError("result is missing string field 'failure_reason'")
    if not isinstance(payload.get("next_action"), str):
        raise MalformedResponseError("result is missing string field 'next_action'")
    failure = failure_detail(payload, label="result")
    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "run_id": run_id,
        "parent_goal_id": payload["parent_goal_id"],
        "parent_step_id": payload["parent_step_id"],
        "execution_status": execution_status,
        "execution_terminal": execution_status in TERMINAL_EXECUTION_STATUSES,
        "change_acceptance": change_acceptance,
        "verification_complete": verification_complete,
        "next_action": payload["next_action"],
        **failure,
        "accepted_artifact": accepted_artifact(payload),
    }


__all__ = [
    "ACCEPTANCE_ACCEPTED",
    "EXECUTION_STATES",
    "EXECUTION_STATUSES",
    "FAILURE_CLASSES",
    "LIVE_EXECUTION_STATES",
    "NOT_EVALUATED",
    "RESULT_ACCEPTANCE_VALUES",
    "STATUS_ACCEPTANCE_VALUES",
    "TERMINAL_EXECUTION_STATES",
    "TERMINAL_EXECUTION_STATUSES",
    "MalformedResponseError",
    "accepted_artifact",
    "cli_error_detail",
    "failure_detail",
    "lifecycle",
    "parse_response",
    "validate_result_shape",
    "validate_status_shape",
]
