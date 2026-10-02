"""Agent tools for the AO CodeRun adapter.

Five tools, one per AO operation: ``submit``, ``status``, ``result``,
``cancel``, and ``events``. They are thin wrappers over
:class:`~raven_ao.adapter.CodeRunAdapter` -- the adapter owns the decisions,
these only marshal arguments and render the adapter's structured answer as JSON
for the model.

All five are controlled by ``request_id``, which is what AO's CLI takes. AO
publishes a ``run_id`` on the result only, so a tool caller that has one may pass
it, but it is never required to ask for status, a result, a cancellation, or the
event log.

Each returns a JSON string, including on failure, so the model always receives
the adapter's fields verbatim rather than a prose summary that could imply a
verdict the adapter never made.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from raven.contracts.tool import Tool
from raven_ao.adapter import AdapterError, CodeRunAdapter
from raven_ao.config import ConfigError, resolve_config

logger = logging.getLogger("raven_ao")

_RESULT_HINT = (
    "An accepted commit or patch appears only when agent-orchestrator reports "
    "change_acceptance=accepted with a concrete artifact and "
    "verification_complete=true."
)

_REQUEST_ID_HINT = (
    "Pass request_id, which submit returns and which controls the run. run_id "
    "is optional: AO publishes it on the result, and it is never required."
)


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)


class _AdapterTool(Tool):
    """One AO operation exposed to the model."""

    _action: str = ""

    def __init__(self, adapter: CodeRunAdapter) -> None:
        self._adapter = adapter

    async def execute(self, **kwargs: Any) -> str:
        action = self._action
        try:
            handler: Callable[..., Any] = getattr(self._adapter, action)
            return _render(await handler(**kwargs))
        except (AdapterError, ConfigError) as exc:
            return _render({"ok": False, "outcome": "invalid_request", "error": str(exc)})

    @property
    def parameters(self) -> dict[str, Any]:
        return _SCHEMAS[self._action]


class SubmitTool(_AdapterTool):
    """Launch a CodeRun for a Raven coding step."""

    _action = "submit"

    @property
    def name(self) -> str:
        return "ao_code_run_submit"

    @property
    def description(self) -> str:
        return (
            "Submit a coding task to agent-orchestrator (AO) as a CodeRun. Pass the "
            "CodeRunRequest as JSON plus parent_goal_id and parent_step_id "
            "identifying this step. The adapter injects the stable request_id and "
            "both parent identifiers, records the mapping durably, and returns AO's "
            "submission state. Calling submit again for the same step reconnects to "
            "the existing run instead of launching a duplicate; a changed request is "
            "reported as changed and is not relaunched."
        )


class StatusTool(_AdapterTool):
    """Read AO's durable status fields."""

    _action = "status"

    @property
    def name(self) -> str:
        return "ao_code_run_status"

    @property
    def description(self) -> str:
        return (
            "Read agent-orchestrator's durable status for a submitted CodeRun. "
            + _REQUEST_ID_HINT
            + " Returns AO's own fields: execution_state (queued, running, "
            "cancel_requested, terminal, blocked, failed, cancelled), "
            "execution_status, and change_acceptance (pending, accepted, "
            "rejected, or not_evaluated while a live run has judged nothing yet)."
        )


class ResultTool(_AdapterTool):
    """Read a formal CodeRunResult."""

    _action = "result"

    @property
    def name(self) -> str:
        return "ao_code_run_result"

    @property
    def description(self) -> str:
        return (
            "Get the formal result of an agent-orchestrator CodeRun. "
            + _REQUEST_ID_HINT
            + " Returns change_acceptance (pending, accepted, rejected), "
            "execution_status (pending, running, completed, blocked, failed, "
            "cancelled), the AO run_id, and failure_class/failure_reason when the "
            "run failed. " + _RESULT_HINT
        )


class CancelTool(_AdapterTool):
    """Request cancellation of a CodeRun."""

    _action = "cancel"

    @property
    def name(self) -> str:
        return "ao_code_run_cancel"

    @property
    def description(self) -> str:
        return (
            "Request cancellation of an agent-orchestrator CodeRun and return AO's "
            "durable state. " + _REQUEST_ID_HINT
            + " Pass an optional reason. The run stays nonterminal in Raven until a "
            "later status or result confirms execution_status=cancelled; nothing is "
            "retried automatically."
        )


class EventsTool(_AdapterTool):
    """Consume one incremental page of AO's durable event log."""

    _action = "events"

    @property
    def name(self) -> str:
        return "ao_code_run_events"

    @property
    def description(self) -> str:
        return (
            "Read the next page of agent-orchestrator's durable CodeRun event log. "
            + _REQUEST_ID_HINT
            + " The adapter resumes from its own stored cursor, so calling this "
            "again returns only events after the last one consumed, and the cursor "
            "survives a restart. Pass after_seq to resume from a specific sequence; "
            "it is refused if it is behind the stored cursor. Returns the events as "
            "a projection of what AO published. Events never decide acceptance or "
            "terminal state: read result for the verdict."
        )


_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "The CodeRunRequest payload. The adapter injects request_id, "
        "parent_goal_id and parent_step_id, and refuses a request that supplies a "
        "conflicting value for any of them."
    ),
}

# request_id is the control key for every AO operation; run_id is optional
# because AO publishes it on the result only.
_IDENTITY_PROPERTIES: dict[str, Any] = {
    "request_id": {"type": "string", "description": "The request id submit returned."},
    "run_id": {
        "type": "string",
        "description": "Optional AO run id, when one has already been read from a result.",
    },
}

_SCHEMAS: dict[str, dict[str, Any]] = {
    "submit": {
        "type": "object",
        "properties": {
            "parent_goal_id": {"type": "string", "description": "Parent goal identifier."},
            "parent_step_id": {"type": "string", "description": "Parent step identifier."},
            "request": _REQUEST_SCHEMA,
        },
        "required": ["parent_goal_id", "parent_step_id", "request"],
    },
    "status": {
        "type": "object",
        "properties": dict(_IDENTITY_PROPERTIES),
    },
    "result": {
        "type": "object",
        "properties": dict(_IDENTITY_PROPERTIES),
    },
    "cancel": {
        "type": "object",
        "properties": {
            **_IDENTITY_PROPERTIES,
            "reason": {"type": "string", "description": "Why the run is being cancelled."},
        },
    },
    "events": {
        "type": "object",
        "properties": {
            **_IDENTITY_PROPERTIES,
            "after_seq": {
                "type": "integer",
                "description": (
                    "Resume strictly after this event sequence. Omit it to "
                    "continue from the stored cursor."
                ),
            },
        },
    },
}

def make_adapter(ctx: Any) -> CodeRunAdapter | None:
    """Shared factory body: resolve config and build the adapter.

    Returns ``None`` when the plugin is not configured, so the host declines to
    register tools that could never succeed. The plugin stays optional: Raven
    works without it.
    """
    try:
        config = resolve_config(ctx)
    except ConfigError as exc:
        logger.info("agent-orchestrator plugin not registered: %s", exc)
        return None
    return CodeRunAdapter(config)


def _make_tool(ctx: Any, tool_type: type[_AdapterTool]) -> Tool | None:
    """Build one manifest-contributed Raven tool."""
    adapter = make_adapter(ctx)
    if adapter is None:
        return None
    return tool_type(adapter)


def make_submit_tool(ctx: Any) -> Tool | None:
    return _make_tool(ctx, SubmitTool)


def make_status_tool(ctx: Any) -> Tool | None:
    return _make_tool(ctx, StatusTool)


def make_result_tool(ctx: Any) -> Tool | None:
    return _make_tool(ctx, ResultTool)


def make_cancel_tool(ctx: Any) -> Tool | None:
    return _make_tool(ctx, CancelTool)


def make_events_tool(ctx: Any) -> Tool | None:
    return _make_tool(ctx, EventsTool)


__all__ = [
    "CancelTool",
    "EventsTool",
    "ResultTool",
    "StatusTool",
    "SubmitTool",
    "make_adapter",
    "make_cancel_tool",
    "make_events_tool",
    "make_result_tool",
    "make_status_tool",
    "make_submit_tool",
]
