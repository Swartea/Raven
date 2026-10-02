"""Plugin configuration for the AO CodeRun adapter.

All three settings the adapter needs -- the AO executable, the AO runs
directory, and the durable mapping directory -- arrive through the manifest's
``[plugin.config_schema]``, so an operator configures them under
``plugins.config["agent-orchestrator"]`` and the host type-checks them at
activation. Nothing here reads environment variables, global config, or a
credential store, and the plugin stays optional: a missing or partial
configuration disables the tools rather than breaking Raven.

``executable`` is the argv *prefix* AO is reached through, because AO is a
module rather than a standalone binary:

```toml
executable = ["python3", "-m", "orchestrator.cli"]
```

A bare string is still accepted and is passed as a single argv element, so a
configured path containing spaces reaches AO intact and no shell ever
interprets it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raven_ao.client import normalize_executable

DEFAULT_MAPPING_DIRNAME = "agent-orchestrator"


@dataclass(frozen=True)
class AdapterConfig:
    """Resolved adapter settings."""

    executable: tuple[str, ...]
    runs_dir: Path
    mapping_dir: Path
    working_dir: Path | None = None
    timeout: float = 600.0


class ConfigError(RuntimeError):
    """The plugin is not configured well enough to be usable."""


def _config_mapping(ctx: Any) -> dict[str, Any]:
    """Read the plugin's config slice off whatever context object was passed.

    The tool factory is called with the host's plugin context, whose config
    accessor has moved between releases; accept the attribute and the mapping
    form so a host change does not break activation.
    """
    for attr in ("config", "plugin_config", "settings"):
        value = getattr(ctx, attr, None)
        if isinstance(value, dict):
            return value
    if isinstance(ctx, dict):
        return ctx
    return {}


def _default_mapping_dir(configured: dict[str, Any]) -> Path:
    base = configured.get("mapping_dir")
    if isinstance(base, str) and base.strip():
        return Path(base).expanduser()
    # No explicit directory: derive one from the AO runs directory when given,
    # so the mapping lives beside the runs it describes.
    runs = configured.get("runs_dir")
    if isinstance(runs, str) and runs.strip():
        return Path(runs).expanduser() / DEFAULT_MAPPING_DIRNAME
    raise ConfigError(
        "agent-orchestrator plugin requires 'mapping_dir' (or 'runs_dir') "
        "in plugins.config['agent-orchestrator']"
    )


def _configured_executable(ctx: Any, configured: dict[str, Any]) -> str | list[str]:
    """Read the AO command as a string or an argv list.

    A bare string naming a program is allowed for convenience, but a list is
    the honest shape: AO is driven as ``python -m orchestrator.cli``, which
    needs more than one argv element, and nothing here is ever shell-split.
    """
    value = configured.get("executable")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(part, str) for part in value
    ):
        return list(value)
    raise ConfigError(
        "agent-orchestrator plugin requires 'executable' (a string or a list of "
        "argv elements, e.g. [\"python3\", \"-m\", \"orchestrator.cli\"]) in "
        "plugins.config['agent-orchestrator']"
    )


def resolve_config(ctx: Any) -> AdapterConfig:
    """Build an :class:`AdapterConfig` from the plugin's config slice.

    Raises :class:`ConfigError` when the executable or a directory is missing,
    which is the signal for the factory to decline registration.
    """
    configured = _config_mapping(ctx)
    executable = normalize_executable(_configured_executable(ctx, configured))
    runs_dir = configured.get("runs_dir")
    if not isinstance(runs_dir, str) or not runs_dir.strip():
        raise ConfigError(
            "agent-orchestrator plugin requires 'runs_dir' in "
            "plugins.config['agent-orchestrator']"
        )
    timeout = configured.get("timeout")
    timeout_value = float(timeout) if isinstance(timeout, (int, float)) else 600.0
    working_dir = configured.get("working_dir")
    if working_dir is not None and (not isinstance(working_dir, str) or not working_dir.strip()):
        raise ConfigError("agent-orchestrator 'working_dir' must be a non-empty string")
    return AdapterConfig(
        executable=executable,
        runs_dir=Path(runs_dir).expanduser(),
        mapping_dir=_default_mapping_dir(configured),
        working_dir=Path(working_dir).expanduser() if working_dir else None,
        timeout=timeout_value,
    )


__all__ = ["AdapterConfig", "ConfigError", "DEFAULT_MAPPING_DIRNAME", "resolve_config"]
