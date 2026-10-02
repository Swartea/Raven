"""Durable local mapping from a Raven coding step to its AO CodeRun.

One file per ``request_id`` under the configured mapping directory, holding the
parent identifiers, the request fingerprint, the AO ``run_id`` once AO has
published one, and the highest AO event sequence Raven has consumed. The store
is the adapter's *only* memory: it holds no AO task state, no attempt DAG, no
execution progress, and no run history. Everything about how the run is going is
asked of AO; this file only answers "which ``request_id`` controls this Raven
step, and how far into AO's event log have I already read?".

``request_id`` is the control key on every AO operation, so a record that has no
``run_id`` yet is still complete: ``run_id`` is a label AO publishes later, not
a prerequisite for driving the run.

``last_seen_event_seq`` is a *consumer cursor*, not run state. AO owns the event
log and is the only source of what happened; Raven stores nothing but the
position it has read to, so a restarted process resumes with a request for
sequences strictly greater than the cursor and neither re-reads nor skips an
event. The cursor is never allowed to move backwards, and it is advanced only
after a validated page is accepted -- an unreadable page leaves it untouched.

Three properties matter and are all tested:

*Atomicity* -- a record is written to a temp file in the same directory and
installed with :func:`os.replace`, so a reader never observes a half-written
record, and a crash mid-write leaves the previous record intact.

*Cross-process safety* -- mutating operations take an exclusive
:func:`fcntl.flock` on a per-record lock file, so two Raven processes racing on
the same parent step serialize instead of interleaving writes.

*Monotonic cursor advance* -- :meth:`MappingStore.advance_event_cursor` performs
its read-modify-write under that same lock and refuses any sequence that is not
strictly greater than the stored cursor, so concurrent consumers of one request
can only ever move the cursor forward.
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
#: AO has explicitly reported that the request behind this mapping no longer
#: exists. This is an observable, recoverable condition -- not a failure and not
#: a verdict about the run -- and it is recorded durably so a later submit does
#: not blindly relaunch a request AO has already forgotten.
STATE_STALE_MAPPING: str = "stale_mapping"

TERMINAL_RAVEN_STATES: frozenset[str] = frozenset()

#: The cursor value that precedes every AO event. AO numbers its durable event
#: log from ``0`` and its ``code-run-events --after-seq`` default is ``-1``, so a
#: fresh (or legacy, cursor-less) reader must start at ``-1`` to be offered event
#: ``0``. Starting at ``0`` would silently skip the first event of every run.
CURSOR_BEFORE_FIRST_EVENT: int = -1


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
    last_seen_event_seq: int = CURSOR_BEFORE_FIRST_EVENT
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

    def _control_lock_for(self, request_id: str) -> Path:
        # Deliberately a *different* file from the record lock. The record lock
        # guards one read-modify-write of the mapping; the control lock guards a
        # whole AO submit/control call sequence that takes the record lock
        # several times inside it. Reusing one file would make the nested
        # acquisition self-deadlock, since flock is held per open description.
        safe = request_id.replace(os.sep, "_").replace("/", "_")
        return self._root / f"{safe}.control.lock"

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

    def try_acquire_control(self, request_id: str) -> tuple[int, bool]:
        """Try to take this request's control lock without blocking.

        Returns ``(fd, acquired)``. Non-blocking on purpose: the caller holds an
        event loop and drives AO asynchronously, and a blocking :func:`flock`
        would freeze that loop for every other coroutine while it waits -- a
        waiter must be able to keep running the lock holder. The caller releases
        with :meth:`release_control` and may poll until it succeeds.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._control_lock_for(request_id), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return -1, False
        return fd, True

    def release_control(self, fd: int) -> None:
        """Release a lock file descriptor taken by :meth:`try_acquire_control`."""
        if fd < 0:
            return
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
        if known.get("last_seen_event_seq") is None:
            # A record written before the cursor existed has read nothing, and
            # "nothing read" is -1 rather than 0: AO's first event is sequence 0,
            # so a 0 default would skip it forever.
            known["last_seen_event_seq"] = CURSOR_BEFORE_FIRST_EVENT
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

    def advance_event_cursor(
        self, request_id: str, sequence: int, *, run_id: str | None = None
    ) -> RunMapping | None:
        """Move the event cursor forward to ``sequence``, never backwards.

        The read-modify-write happens under the record's exclusive lock, so two
        Raven processes consuming the same request serialize: the loser reads
        the winner's advanced cursor and, if its own sequence is no longer
        strictly greater, the advance is refused instead of rewinding the
        cursor and causing the winner's range to be replayed.

        ``run_id`` is a run identity AO may publish for the first time on the
        very page that advances the cursor. It is written in the *same* locked
        read-modify-write rather than as a separate update, because the pair is
        one event position: a crash between two writes could otherwise leave a
        durable cursor attributed to no run, or a run whose cursor never moved.
        An existing ``run_id`` is never overwritten, and is never replaced by a
        value the page did not publish.

        Returns the record as it now stands, or ``None`` when no mapping exists
        (there is nothing to advance). Refusal is not an error: a caller that
        loses the race simply reports the cursor that is already ahead.
        """
        with self._locked(request_id):
            existing = self._read_unlocked(request_id)
            if existing is None:
                return None
            learned_run_id = (
                isinstance(run_id, str) and run_id.strip() and existing.run_id is None
            )
            if sequence <= existing.last_seen_event_seq and not learned_run_id:
                return existing
            if sequence > existing.last_seen_event_seq:
                existing.last_seen_event_seq = sequence
            if learned_run_id:
                existing.run_id = run_id
            existing.updated_at = _now()
            self._write_atomic(existing)
            return existing


class CapacityGuard:
    """Durable, cross-process cap on concurrently active AO CodeRuns.

    Raven's *outer* concurrency policy: how many AO CodeRuns this Raven
    deployment will have in flight at once, independent of whatever AO's own
    internal dispatch parallelism happens to be. Capacity is claimed before a
    launch and released only once AO itself has confirmed the run can no longer
    consume it -- an authoritative terminal status/result, or AO confirming the
    request no longer exists. A run that is merely quiet, or that Raven has not
    yet been told anything about, keeps its slot.

    The guard is a set of empty marker files under ``active`` inside the mapping
    directory, one per active ``request_id``, plus a single guard lock. Because
    the count is derived from the directory rather than from a counter in
    memory, a Raven process that dies holding a slot does not permanently inflate
    the count in a way another process cannot see or reclaim.
    """

    #: Marker files are named after the request id, which is adapter-derived hex.
    _SUFFIX: str = ".active"

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._root = Path(directory).expanduser()
        self._active = self._root / "active"
        self._lock_path = self._root / "capacity.lock"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def active_dir(self) -> Path:
        return self._active

    def _marker_for(self, request_id: str) -> Path:
        safe = request_id.replace(os.sep, "_").replace("/", "_")
        return self._active / f"{safe}{self._SUFFIX}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._root.mkdir(parents=True, exist_ok=True)
        self._active.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def active_count(self) -> int:
        """How many CodeRuns currently hold a slot, across all processes."""
        if not self._active.exists():
            return 0
        return sum(1 for _ in self._active.glob(f"*{self._SUFFIX}"))

    def active_request_ids(self) -> list[str]:
        if not self._active.exists():
            return []
        return sorted(
            path.name[: -len(self._SUFFIX)] for path in self._active.glob(f"*{self._SUFFIX}")
        )

    def acquire(self, request_id: str, limit: int) -> tuple[bool, int]:
        """Claim a slot for ``request_id`` when fewer than ``limit`` are active.

        Returns ``(acquired, active_count)``. Re-acquiring a slot this request
        already holds is a no-op that still succeeds, so a resubmit or reconnect
        of the same run can never be blocked by its own earlier claim.
        """
        if limit < 1:
            return False, self.active_count()
        with self._locked():
            marker = self._marker_for(request_id)
            if marker.exists():
                return True, self.active_count()
            if self.active_count() >= limit:
                return False, self.active_count()
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return True, self.active_count()

    def holds(self, request_id: str) -> bool:
        return self._marker_for(request_id).exists()

    def release(self, request_id: str) -> bool:
        """Give back ``request_id``'s slot. Returns whether one was held.

        Releasing a slot that is not held is not an error: the run is no longer
        active either way, and the caller's intent is satisfied.
        """
        with self._locked():
            try:
                self._marker_for(request_id).unlink()
                return True
            except FileNotFoundError:
                return False


__all__ = [
    "CURSOR_BEFORE_FIRST_EVENT",
    "CapacityGuard",
    "MappingStore",
    "RunMapping",
    "STATE_CANCEL_REQUESTED",
    "STATE_RESERVED",
    "STATE_STALE_MAPPING",
    "STATE_SUBMITTED",
    "derive_request_id",
    "fingerprint_request",
]
