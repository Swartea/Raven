"""Durable Raven -> agent-orchestrator (AO) CodeRun plugin adapter.

Bridges a Raven coding step to an AO CodeRun over AO's ``code-run-*`` CLI,
exposing exactly four operations: submit, status, result, cancel. Every one of
them is controlled by ``request_id``. Raven holds only a durable
step-to-request mapping; all run state and every acceptance verdict come from AO.
"""

from raven_ao.adapter import AdapterError, CodeRunAdapter
from raven_ao.client import AoCliClient, CliInvocationError
from raven_ao.config import AdapterConfig, ConfigError, resolve_config
from raven_ao.contract import (
    FAILURE_CLASSES,
    RESULT_ACCEPTANCE_VALUES,
    STATUS_ACCEPTANCE_VALUES,
    MalformedResponseError,
)
from raven_ao.mapping import MappingStore, RunMapping, derive_request_id

__all__ = [
    "AoCliClient",
    "AdapterConfig",
    "AdapterError",
    "CliInvocationError",
    "CodeRunAdapter",
    "ConfigError",
    "FAILURE_CLASSES",
    "MalformedResponseError",
    "MappingStore",
    "RESULT_ACCEPTANCE_VALUES",
    "RunMapping",
    "STATUS_ACCEPTANCE_VALUES",
    "derive_request_id",
    "resolve_config",
]
