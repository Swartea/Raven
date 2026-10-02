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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    STATE_CANCEL_REQUESTED,
    STATE_RESERVED,
    STATE_SUBMITTED,
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


class AdapterError(RuntimeError):
    """A user-facing adapter failure (bad arguments, no such run, bad config)."""


def _stamp() -> str:
    from datetime import UTC, datetime

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

    @property
    def store(self) -> MappingStore:
        return self._store

    @property
    def client(self) -> AoCliClient:
        return self._client

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
        candidate = RunMapping(
            request_id=request_id,
            parent_goal_id=parent_goal_id,
            parent_step_id=parent_step_id,
            request_fingerprint=fingerprint_request(
                effective_request(parent_goal_id, parent_step_id, request, request_id)
            ),
        )
        # Reserve first: a crash after the AO call must not orphan a run, and a
        # concurrent process must not race us into a second launch.
        existing, created = self._store.reserve(candidate)
        if not created:
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
            return {
                "ok": False,
                "outcome": "launch_failed",
                "request_id": existing.request_id,
                "parent_goal_id": parent_goal_id,
                "parent_step_id": parent_step_id,
                "run_id": None,
                "error": (
                    f"AO CLI 'code-run-submit' exited {cli_result.exit_code}: "
                    f"{cli_error_detail(cli_result.stdout, cli_result.stderr)[:400]}"
                ),
                "detail": (
                    "The AO CLI refused the submission. No run was launched; "
                    "reconnect with status/result on this request_id."
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
            payload["request_changed"] = True
            payload["detail"] = (
                "An existing CodeRun is already mapped to this parent step with a "
                "different request. The existing run was reused; no second run "
                "was launched."
            )
        if existing.state == STATE_RESERVED and existing.submitted_at is None:
            try:
                status_result = await self._client.status(existing.request_id)
            except CliInvocationError as exc:
                return _launch_unconfirmed(existing, exc)
            if status_result.ok:
                try:
                    status_payload = parse_response(status_result.stdout)
                    shaped = validate_status_shape(status_payload)
                    if shaped["request_id"] != existing.request_id:
                        raise MalformedResponseError("status answered for a different request_id")
                except MalformedResponseError as exc:
                    return _launch_unconfirmed(existing, exc)
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
                return recovered
            if not _is_missing_request(status_result):
                return _launch_unconfirmed(
                    existing,
                    CliInvocationError(
                        "could not determine whether the reserved request reached AO: "
                        + cli_error_detail(status_result.stdout, status_result.stderr)[:400]
                    ),
                )

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

            retry_request = effective_request(
                candidate.parent_goal_id,
                candidate.parent_step_id,
                request,
                candidate.request_id,
            )
            try:
                retried = await self._client.submit(retry_request)
            except CliInvocationError as exc:
                return _launch_unconfirmed(existing, exc)
            if not retried.ok:
                # Another process may have won the AO request-id election.
                try:
                    latest = await self._client.status(existing.request_id)
                except CliInvocationError as exc:
                    return _launch_unconfirmed(existing, exc)
                if not latest.ok:
                    return _launch_unconfirmed(
                        existing,
                        CliInvocationError(
                            "AO did not confirm recovery after submit: "
                            + cli_error_detail(retried.stdout, retried.stderr)[:400]
                        ),
                    )
                retried = latest
            try:
                status_payload = parse_response(retried.stdout)
                shaped = validate_status_shape(status_payload)
                if shaped["request_id"] != existing.request_id:
                    raise MalformedResponseError("recovery answered for a different request_id")
            except MalformedResponseError as exc:
                return _launch_unconfirmed(existing, exc)
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
            return recovered
        return payload

    # -- mapping resolution ---------------------------------------------
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

    def _iter_mappings(self):
        root = self._store.root
        if not root.exists():
            return
        for path in sorted(root.glob("*.json")):
            mapping = self._store.get(path.stem)
            if mapping is not None:
                yield mapping

    # -- status ---------------------------------------------------------
    async def status(
        self, request_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """Return AO's durable status, driven by ``request_id``."""
        mapping = self._resolve(request_id, run_id)
        try:
            payload = _formal(
                await self._client.status(mapping.request_id), "code-run-status"
            )
            shaped = validate_status_shape(payload)
            if shaped["request_id"] != mapping.request_id:
                raise MalformedResponseError("status answered for a different request_id")
        except (CliInvocationError, MalformedResponseError) as exc:
            return _cli_failure(mapping, "code-run-status", exc)

        run_id = self._record_run_id(mapping, payload)
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
            payload = _formal(
                await self._client.result(mapping.request_id), "code-run-result"
            )
            shaped = validate_result_shape(payload)
        except (CliInvocationError, MalformedResponseError) as exc:
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
            payload = _formal(
                await self._client.cancel(mapping.request_id, reason), "code-run-cancel"
            )
            shaped = validate_status_shape(payload)
            if shaped["request_id"] != mapping.request_id:
                raise MalformedResponseError("cancel answered for a different request_id")
        except (CliInvocationError, MalformedResponseError) as exc:
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
