"""Direct ``argv`` invocation of the agent-orchestrator ``code-run-*`` CLI.

The four operations map onto AO's own subcommands, and the command surface is
transcribed from AO rather than assumed:

===========================  =========================================
``code-run-submit``          ``--request PATH`` plus ``--runs-dir``
``code-run-status``          positional ``REQUEST_ID``
``code-run-result``          positional ``REQUEST_ID``
``code-run-cancel``          positional ``REQUEST_ID`` plus ``--reason``
===========================  =========================================

There is no ``--json`` flag, no generic ``submit``/``status`` command, and no
``run_id`` argument. Control is by ``request_id`` on every operation *including*
status, result and cancel, and AO answers all three from durable state, so a
process that did not submit the run drives it exactly like the one that did.

``code-run-submit`` reads the whole ``CodeRunRequest`` from a **file**, not from
stdin, so the request is written to a private temporary file and passed by path.

The AO executable is always run as an argument vector with ``shell=False``. No
string is ever handed to a shell, so a path, a prompt, or a patch carried in a
CodeRunRequest cannot be reinterpreted as shell syntax. ``executable`` is the
argv *prefix* AO is reached through -- typically ``["python3", "-m",
"orchestrator.cli"]`` -- because AO is a module, not a standalone binary.

A non-zero exit is *not* an answer: it is a failed invocation. AO reports a
refused command on stdout as ``{"error": {...}}`` with exit status 2, and the
caller must treat either as a CLI failure, never as a verdict about the run.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

DEFAULT_TIMEOUT_SECONDS: Final = 600.0

#: AO's exact subcommand names. Anything else is not this CLI.
COMMAND_SUBMIT: Final = "code-run-submit"
COMMAND_STATUS: Final = "code-run-status"
COMMAND_RESULT: Final = "code-run-result"
COMMAND_CANCEL: Final = "code-run-cancel"

AO_COMMANDS: Final = frozenset(
    {COMMAND_SUBMIT, COMMAND_STATUS, COMMAND_RESULT, COMMAND_CANCEL}
)


class CliInvocationError(RuntimeError):
    """The AO CLI could not be run, or exited non-zero.

    Carries no verdict: the caller must not read a run's acceptance out of this.
    """


@dataclass(frozen=True)
class CliResult:
    """Raw outcome of one CLI invocation."""

    exit_code: int
    stdout: str
    stderr: str
    argv: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def normalize_executable(executable: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize the configured AO command into an argv prefix.

    Accepts a sequence of argv elements or a single string. A single string is
    kept as **one** argv element rather than shell-split: a configured path
    containing a space must reach AO intact, and no shell ever interprets it.
    An operator who needs the module form configures the list.
    """
    if isinstance(executable, str):
        candidate = executable.strip()
        return (candidate,) if candidate else ()
    tokens = tuple(str(part) for part in executable if str(part).strip())
    return tokens


def resolve_executable(executable: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve the argv prefix, expanding a bare program name via ``PATH``.

    Only the first element is looked up: everything after it is an argument AO
    itself consumes (``-m orchestrator.cli``), and must not be resolved.
    """
    tokens = normalize_executable(executable)
    if not tokens:
        return tokens
    program, *rest = tokens
    candidate = Path(program).expanduser()
    if candidate.is_absolute() or os.sep in program:
        return (str(candidate), *rest)
    return (shutil.which(program) or program, *rest)


class AoCliClient:
    """Thin, shell-free wrapper around AO's ``code-run-*`` CLI."""

    def __init__(
        self,
        executable: str | Sequence[str],
        *,
        runs_dir: str | Path | None = None,
        working_dir: str | Path | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._executable = normalize_executable(executable)
        if not self._executable:
            raise CliInvocationError("no AO executable configured")
        self._runs_dir = str(runs_dir) if runs_dir is not None else None
        self._working_dir = str(working_dir) if working_dir is not None else None
        self._timeout = timeout

    @property
    def executable(self) -> tuple[str, ...]:
        return self._executable

    @property
    def runs_dir(self) -> str | None:
        return self._runs_dir

    @staticmethod
    def build_argv(
        executable: Sequence[str], command: str, *args: str
    ) -> list[str]:
        """Compose the exact argv for one AO subcommand.

        The subcommand comes first, positional arguments next, then flags. The
        order is asserted in the tests, because AO's ``request_id`` is a
        positional argument and must not be passed with a flag.
        """
        if command not in AO_COMMANDS:
            raise CliInvocationError(f"unknown AO command: {command!r}")
        return [*executable, command, *args]

    def _command_argv(self, command: str, *args: str) -> list[str]:
        argv = self.build_argv(self._executable, command, *args)
        if self._runs_dir:
            argv += ["--runs-dir", self._runs_dir]
        return argv

    async def _invoke(self, argv: list[str], command: str) -> CliResult:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
        except FileNotFoundError as exc:
            raise CliInvocationError(
                f"AO executable not found: {argv[0]}"
            ) from exc
        except OSError as exc:
            raise CliInvocationError(f"could not start AO CLI: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            process.kill()
            await process.wait()
            raise CliInvocationError(
                f"AO CLI '{command}' timed out after {self._timeout}s"
            ) from exc

        return CliResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            argv=tuple(argv),
        )

    async def submit(self, request: dict[str, Any]) -> CliResult:
        """Submit one coding package.

        AO reads the ``CodeRunRequest`` from the file named by ``--request``, so
        the request is serialized into a private temporary file that is removed
        once AO has been invoked. Nothing about the request passes through argv.
        """
        argv = self._command_argv(COMMAND_SUBMIT)
        handle, request_path = tempfile.mkstemp(prefix="raven-ao-request-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(request, stream, ensure_ascii=False, indent=2, sort_keys=True)
            argv += ["--request", request_path]
            return await self._invoke(argv, COMMAND_SUBMIT)
        finally:
            try:
                os.unlink(request_path)
            except OSError:
                pass

    async def status(self, request_id: str) -> CliResult:
        return await self._invoke(self._command_argv(COMMAND_STATUS, request_id), COMMAND_STATUS)

    async def result(self, request_id: str) -> CliResult:
        return await self._invoke(self._command_argv(COMMAND_RESULT, request_id), COMMAND_RESULT)

    async def cancel(self, request_id: str, reason: str | None = None) -> CliResult:
        argv = self._command_argv(COMMAND_CANCEL, request_id)
        if reason:
            argv += ["--reason", reason]
        return await self._invoke(argv, COMMAND_CANCEL)


__all__ = [
    "AO_COMMANDS",
    "COMMAND_CANCEL",
    "COMMAND_RESULT",
    "COMMAND_STATUS",
    "COMMAND_SUBMIT",
    "AoCliClient",
    "CliInvocationError",
    "CliResult",
    "DEFAULT_TIMEOUT_SECONDS",
    "normalize_executable",
    "resolve_executable",
]
