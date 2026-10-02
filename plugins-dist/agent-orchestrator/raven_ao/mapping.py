"""Durable local mapping from a Raven coding step to its AO CodeRun.

One file per ``request_id`` under the configured mapping directory, holding the
parent identifiers, the request fingerprint, and the AO ``run_id`` once AO has
published one. The store is the adapter's *only* memory: it holds no AO task
state, no attempt DAG, and no execution progress. Everything about how the run
is going is asked of AO; this file only answers "which ``request_id`` controls
this Raven step?".

``request_id`` is the control key on every AO operation, so a record that has no
``run_id`` yet is still complete: ``run_id`` is a label AO publishes later, not
a prerequisite for driving the run.

Two properties matter and are both tested:

*Atomicity* -- a record is written to a temp file in the same directory and
installed with :func:`os.replace`, so a reader never observes a half-written
record, and a crash mid-write leaves the previous record intact.

*Cross-process safety* -- mutating operations take an exclusive
:func:`fcntl.flock` on a per-record lock file, so two Raven processes racing on
the same parent step serialize instead of interleaving writes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

# Raven-side lifecycle of the mapping. These are *Raven's* states, describing
# the adapter's own bookkeeping only -- never the run's execution state, which
# AO owns and which the adapter always reads fresh from AO.
STATE_RESERVED: str = "reserved"
STATE_SUBMITTED: str = "submitted"
STATE_CANCEL_REQUESTED: str = "cancel_requested"

TERMINAL_RAVEN_STATES: frozenset[str] = frozenset()


class MappingCorruptError(RuntimeError):
    """A durable mapping exists but cannot be trusted or replaced safely."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def derive_request_id(parent_goal_id: str, parent_step_id: str) -> str:
    """Derive a stable AO ``request_id`` from the parent Raven identifiers.

    Deterministic by construction: the same parent goal and step always yield
    the same id, in this process or any later one, which is what makes a repeat
    submit or a reconnect land on the existing mapping instead of a new run.
    The parent ids are length-prefixed so that no pair of distinct identifiers
    can collide by concatenation (``("a", "bc")`` vs ``("ab", "c")``).
    """
    material = "\x1f".join(
        (str(len(parent_goal_id)), parent_goal_id, str(len(parent_step_id)), parent_step_id)
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"raven-{digest[:32]}"


def fingerprint_request(payload: Any) -> str:
    """Hash a CodeRunRequest so a changed submission is detectable."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class RunMapping:
    """The durable record linking one Raven step to one AO CodeRun."""

    request_id: str
    parent_goal_id: str
    parent_step_id: str
    request_fingerprint: str
    run_id: str | None = None
    state: str = STATE_RESERVED
    cancel_requested: bool = False
    submitted_at: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_run(self) -> bool:
        """Whether AO has published a ``run_id`` for this request yet.

        Informational only. AO controls a run by ``request_id``, so a mapping
        without a ``run_id`` is still fully drivable -- status, result and
        cancel all work -- and must never be treated as "not launched".
        """
        return self.run_id is not None

    @property
    def cancel_pending(self) -> bool:
        """True while a cancel is requested but AO has not yet confirmed it.

        Deliberately nonterminal: Raven keeps reporting the run as in flight
        until a later AO status or result confirms a terminal cancellation, so
        this stays true across the whole ``cancel_requested`` window.
        """
        return self.cancel_requested


class MappingStore:
    """File-backed :class:`RunMapping` store, safe across processes."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._root = Path(directory).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, request_id: str) -> Path:
        # request_id is adapter-derived hex, but keep the filename free of any
        # path separator regardless of who supplies it.
        safe = request_id.replace(os.sep, "_").replace("/", "_")
        return self._root / f"{safe}.json"

    def _lock_for(self, request_id: str) -> Path:
        safe = request_id.replace(os.sep, "_").replace("/", "_")
        return self._root / f"{safe}.lock"

    @contextmanager
    def _locked(self, request_id: str) -> Iterator[None]:
        self._root.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_for(request_id)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_unlocked(self, request_id: str) -> RunMapping | None:
        path = self._path_for(request_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MappingCorruptError(
                f"mapping for request_id {request_id!r} exists but cannot be read; "
                "refusing to overwrite it"
            ) from exc
        if not isinstance(raw, dict):
            raise MappingCorruptError(
                f"mapping for request_id {request_id!r} is not a JSON object; "
                "refusing to overwrite it"
            )
        known = {f: raw.get(f) for f in RunMapping.__dataclass_fields__ if f in raw}
        try:
            return RunMapping(**known)  # type: ignore[arg-type]
        except TypeError as exc:
            raise MappingCorruptError(
                f"mapping for request_id {request_id!r} is missing required fields; "
                "refusing to overwrite it"
            ) from exc

    def _write_atomic(self, mapping: RunMapping) -> None:
        """Install the record atomically via same-directory temp + rename."""
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._path_for(mapping.request_id)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._root), prefix=f".{mapping.request_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(mapping.to_dict(), handle, sort_keys=True, ensure_ascii=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            # Never leave a temp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def get(self, request_id: str) -> RunMapping | None:
        with self._locked(request_id):
            return self._read_unlocked(request_id)

    def find(self, parent_goal_id: str, parent_step_id: str) -> RunMapping | None:
        """Look up by parent identifiers, via the derived id."""
        return self.get(derive_request_id(parent_goal_id, parent_step_id))

    def reserve(self, mapping: RunMapping) -> tuple[RunMapping, bool]:
        """Claim the mapping for this parent step, before any AO launch.

        Returns ``(mapping, created)``. ``created`` is False when a record
        already existed, which is the signal that this is a repeat submit and
        the caller must reuse the existing mapping rather than launching a
        second CodeRun. Reservation happens *first* so a crash between here and
        the AO call cannot produce an untracked run.
        """
        with self._locked(mapping.request_id):
            existing = self._read_unlocked(mapping.request_id)
            if existing is not None:
                return existing, False
            self._write_atomic(mapping)
            return mapping, True

    def update(self, request_id: str, **changes: Any) -> RunMapping | None:
        with self._locked(request_id):
            existing = self._read_unlocked(request_id)
            if existing is None:
                return None
            for key, value in changes.items():
                if key in RunMapping.__dataclass_fields__:
                    setattr(existing, key, value)
            existing.updated_at = _now()
            self._write_atomic(existing)
            return existing


__all__ = [
    "MappingStore",
    "RunMapping",
    "STATE_CANCEL_REQUESTED",
    "STATE_RESERVED",
    "STATE_SUBMITTED",
    "derive_request_id",
    "fingerprint_request",
]
