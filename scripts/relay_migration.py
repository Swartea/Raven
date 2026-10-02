"""Import a Relay repository's execution history into a Raven stint project.

Relay proved nothing in Raven's sense: a Relay `passes: true` is a claim the
legacy harness made about itself, with no Verifier verdict and no evidence
Raven accepts. So nothing here writes `done`, `proven`, or an AO acceptance
field. Legacy completion becomes an open card that a human must unblock, and
every imported outcome is labelled `legacy_execution_evidence`.

The Relay repository is read-only. It is opened, hashed, and read; it is never
written to and no Relay command is invoked, so the source HEAD, worktree, and
file contents are exactly what the manifest describes when the import rejects a
drifted source.

The target is a normal Raven stint project: `.stint/backlog.json` through
`raven.stint.backlog`, `JOURNAL.md` through its append-only helper, and one
durable receipt under `.stint/relay_migration/`. This adds no second progress
database and no new runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "raven.relay-migration"
EVIDENCE_KIND = "legacy_execution_evidence"
UNVERIFIED_BLOCKER = "external:relay-legacy-claim-needs-review"

RECEIPT_DIR = Path(".stint") / "relay_migration"
RECEIPT_FILE = "import_receipt.json"
LEGACY_SPEC_FILE = "LEGACY_RELAY_SPEC.md"

#: Relay kept one directory, then another. Both are read, current first.
FEATURE_PATHS = (
    Path("feature_list.json"),
    Path(".agent_relay") / "features.json",
    Path(".agent_harness") / "features.json",
    Path("features.json"),
)
PROGRESS_PATHS = (
    Path("agent-progress.txt"),
    Path("claude-progress.txt"),
)
OUTCOME_DIRS = (
    Path(".agent_relay") / "session_outcomes",
    Path(".agent_harness") / "session_outcomes",
)
SPEC_PATHS = (Path("project_spec.md"), Path("PROJECT_SPEC.md"))

RELAY_STATUSES = ("pending", "in_progress", "completed", "blocked", "failed")
COMPLETION_CLAIMS = ("completed", "complete", "done", "passed")

_RELATION_PREFIX = re.compile(r"^\s*[-*\d.]+\s*")
_PROGRESS_HEADING = re.compile(r"^\s*(?:#{1,6}\s+)?(session|round|progress|step)\b", re.I)
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_EMBEDDED_ABSOLUTE_PATH = re.compile(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+[\\/]|/)[^\s\"'<>]+")


class MigrationError(RuntimeError):
    """The migration cannot proceed, and the reason is in the message."""


class StaleManifestError(MigrationError):
    """The source moved after the manifest was built; regenerate it."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sanitize_record_paths(value: Any, source: Path) -> tuple[Any, bool]:
    """Keep outcome data while preventing host-specific absolute paths from escaping."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            sanitized[key], item_changed = _sanitize_record_paths(item, source)
            changed |= item_changed
        return sanitized, changed
    if isinstance(value, list):
        sanitized_items = [_sanitize_record_paths(item, source) for item in value]
        return [item for item, _ in sanitized_items], any(changed for _, changed in sanitized_items)
    if not isinstance(value, str):
        return value, False

    text = value
    candidate = Path(text).expanduser()
    windows_absolute = PureWindowsPath(text).is_absolute()
    if candidate.is_absolute() or windows_absolute:
        if windows_absolute and not candidate.is_absolute():
            text = f"<redacted-absolute-path:{PureWindowsPath(text).name or 'path'}>"
        else:
            try:
                relative = candidate.resolve(strict=False).relative_to(source.resolve())
            except (OSError, ValueError):
                text = f"<redacted-absolute-path:{candidate.name or 'path'}>"
            else:
                text = relative.as_posix()
    scrubbed = _EMBEDDED_ABSOLUTE_PATH.sub("<redacted-absolute-path>", text)
    return scrubbed, scrubbed != value


def _atomic_write(path: Path, text: str) -> None:
    """Replace a file in one step so an interrupted import never truncates it.

    The tempfile lives in the destination directory so the final rename cannot
    cross a filesystem boundary and degrade into a copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _git(source: Path, *args: str) -> str | None:
    """Run a read-only git query; None when the source is not a git repository."""
    try:
        git = shutil.which("git")
        if git is None:
            return None
        completed = subprocess.run(
            (git, "-C", str(source), *args),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def sanitize_repository_identity(source: Path) -> str:
    remote = _git(source, "remote", "get-url", "origin")
    if remote:
        remote = re.sub(r"^https?://[^/@]+@", "https://", remote.strip())
        remote = re.sub(r"^[^@]+@([^:]+):", r"\1/", remote)
        name = remote.removesuffix(".git")
    else:
        head = _git(source, "rev-parse", "HEAD") or source.name
        name = f"local-{_sha256(head.encode())[:12]}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned or "relay-source"


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _provenance_line(provenance: dict[str, str]) -> str:
    return (
        f"{provenance['source_kind']} from {provenance['source_repository']} "
        f"@ {provenance['source_head'][:12] or 'unknown'} (migration {provenance['migration_id']})"
    )


def _is_failure(outcome: dict[str, Any]) -> bool:
    status = str(outcome.get("status", "")).lower()
    return status.endswith("failed") or "fail" in status or bool(outcome.get("failure_reason"))


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "passed", "pass")
    return bool(value)


@dataclass
class RelayMigrationManifest:
    """A versioned, self-checking snapshot of one Relay repository."""

    schema_version: int
    migration_id: str
    source_repository: str
    source_head: str
    source_harness_version: str
    source_dirty: bool
    project_spec: str
    work_items: list[dict[str, Any]]
    progress_entries: list[dict[str, Any]]
    session_outcomes: list[dict[str, Any]]
    source_commits: list[dict[str, Any]]
    provenance: dict[str, str]
    checksums: dict[str, str]
    warnings: list[str]
    created_at: str = ""
    raw_progress: str = ""
    input_files: list[str] = field(default_factory=list)
    source_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "schema_version": self.schema_version,
            "migration_id": self.migration_id,
            "created_at": self.created_at,
            "raw_progress": self.raw_progress,
            "input_files": self.input_files,
            "source_status": self.source_status,
            "source": {
                "repository": self.source_repository,
                "head": self.source_head,
                "harness_version": self.source_harness_version,
                "dirty": self.source_dirty,
            },
            "project_spec": self.project_spec,
            "work_items": self.work_items,
            "progress_entries": self.progress_entries,
            "session_outcomes": self.session_outcomes,
            "source_commits": self.source_commits,
            "provenance": self.provenance,
            "checksums": dict(self.checksums),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelayMigrationManifest":
        if not isinstance(data, dict):
            raise MigrationError("manifest is not a JSON object")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MigrationError(
                f"manifest schema_version {version!r} is not supported; this importer reads {SCHEMA_VERSION}"
            )
        if data.get("schema") != MANIFEST_SCHEMA:
            raise MigrationError("manifest is not a Relay migration manifest")
        source = data.get("source") or {}
        if not isinstance(source, dict):
            raise MigrationError("manifest source section is not an object")
        provenance = data.get("provenance")
        if not isinstance(provenance, dict):
            raise MigrationError("manifest provenance is incomplete")
        work_items = data.get("work_items") or []
        entries = data.get("progress_entries") or []
        outcomes = data.get("session_outcomes") or []
        commits = data.get("source_commits") or []
        if not all(isinstance(records, list) for records in (work_items, entries, outcomes, commits)):
            raise MigrationError("manifest records must be JSON arrays")
        if not all(isinstance(item, dict) for records in (work_items, entries, outcomes, commits) for item in records):
            raise MigrationError("manifest records must contain JSON objects")
        return cls(
            schema_version=int(version),
            migration_id=str(data["migration_id"]),
            source_repository=str(source.get("repository", "")),
            source_head=str(source.get("head", "")),
            source_harness_version=str(source.get("harness_version", "unknown")),
            source_dirty=bool(source.get("dirty")),
            project_spec=str(data.get("project_spec", "")),
            work_items=work_items,
            progress_entries=entries,
            session_outcomes=outcomes,
            source_commits=commits,
            provenance={str(k): str(v) for k, v in provenance.items()},
            checksums={str(k): str(v) for k, v in (data.get("checksums") or {}).items()},
            warnings=[str(item) for item in data.get("warnings") or ()],
            created_at=str(data.get("created_at", "")),
            raw_progress=str(data.get("raw_progress", "")),
            input_files=[str(x) for x in data.get("input_files", [])],
            source_status=str(data.get("source_status", "")),
        )

    @classmethod
    def load(cls, path: Path) -> "RelayMigrationManifest":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MigrationError(f"could not read manifest {path}: {error}") from error
        return cls.from_dict(raw)

    def dump(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# --------------------------------------------------------------------------
# Reading the Relay source
# --------------------------------------------------------------------------


def _find_first(source: Path, candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if (source / candidate).is_file():
            return candidate
    return None


def _normalise_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in COMPLETION_CLAIMS:
        return "completed"
    return status if status in RELAY_STATUSES else "pending"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def parse_feature_records(data: Any, *, origin: str) -> list[dict[str, Any]]:
    """Read Relay feature records, including the legacy field spellings.

    Relay's own rule is that `passes: true` outranks `status`, that `steps`
    aliases the acceptance criteria, that a missing status means pending, and
    that an unrecognised status normalises to pending. Reading them any other
    way would silently upgrade a legacy claim into a fact.
    """
    if isinstance(data, dict):
        for key in ("features", "items", "work_items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise MigrationError(f"Relay features at {origin} are not a list of records")

    items: list[dict[str, Any]] = []
    for ordinal, record in enumerate(data):
        if not isinstance(record, dict):
            raise MigrationError(f"Relay feature #{ordinal} at {origin} is not an object")
        feature_id = str(record.get("id") or record.get("feature_id") or "").strip()
        if not feature_id:
            raise MigrationError(f"Relay feature #{ordinal} at {origin} has no id")
        title = str(record.get("title") or record.get("name") or feature_id).strip()
        source_status = str(record.get("status") or "pending").strip()
        status = _normalise_status(source_status)
        passes = _opt_bool(record.get("passes")) is True
        steps = _as_list(record.get("steps") or record.get("acceptance_criteria"))
        items.append(
            {
                "feature_id": feature_id,
                "title": title,
                "relay_status": status,
                "source_status": source_status,
                "claimed_complete": passes or status == "completed",
                "passes_claim": passes,
                "steps": steps,
                "notes": str(record.get("notes") or "").strip(),
                "last_updated": str(record.get("last_updated") or "").strip(),
                "source_ordinal": ordinal,
                "source_file": origin,
                "source_record": dict(record),
                "source_commit": "",
                "source_timestamp": "",
            }
        )
    return items


def parse_progress_text(text: str, *, source_file: str = "") -> list[dict[str, Any]]:
    """Split progress into entries, keeping each one's text exactly as written."""
    entries: list[dict[str, Any]] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if _PROGRESS_HEADING.match(line.rstrip("\r\n")) and current:
            entries.append(_progress_record(len(entries), "".join(current), source_file))
            current = [line]
            continue
        current.append(line)
    if current and "".join(current).strip():
        entries.append(_progress_record(len(entries), "".join(current), source_file))
    return entries


def _progress_record(index: int, raw: str, source_file: str) -> dict[str, Any]:
    return {
        "index": index,
        "raw": raw,
        "source_file": source_file,
        "source_commit": "",
        "source_timestamp": "",
    }


def parse_outcome_record(data: Any, *, source_path: str, source_kind: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise MigrationError(f"session outcome {source_path} is not an object")
    session_id = str(data.get("session_id") or "").strip()
    if not session_id:
        raise MigrationError(f"session outcome {source_path} has no session_id")
    return {
        "session_id": session_id,
        "feature_id": str(data.get("feature_id") or "").strip(),
        "status": str(data.get("status") or "unknown").strip(),
        "evidence_kind": EVIDENCE_KIND,
        "message": str(data.get("message") or ""),
        "prompt_path": str(data.get("prompt_path") or ""),
        "smoke_test_passed": _opt_bool(data.get("smoke_test_passed")),
        "commit_created": _opt_bool(data.get("commit_created")),
        "verification_passed": _opt_bool(data.get("verification_passed")),
        "evaluator_result": str(data.get("evaluator_result") or ""),
        "qa_passed": _opt_bool(data.get("qa_passed")),
        "qa_failure_stage": str(data.get("qa_failure_stage") or ""),
        "qa_results": list(data.get("qa_results") or ()),
        "failure_reason": str(data.get("failure_reason") or ""),
        "source_path": source_path,
        "source_kind": source_kind,
        "source_record": dict(data),
        "source_commit": "",
        "source_timestamp": "",
    }


def read_source(
    source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, str, dict[str, str], list[str]]:
    """Read every Relay input. Returns work items, progress, outcomes, spec, and checksums."""
    source = Path(source).expanduser()
    if not source.is_dir():
        raise MigrationError(f"Relay source {source} is not a directory")
    warnings: list[str] = []
    checksums: dict[str, str] = {}
    raw_progress = ""

    feature_rel = _find_first(source, FEATURE_PATHS)
    if feature_rel is None:
        raise MigrationError("no Relay feature file found; looked for " + ", ".join(str(p) for p in FEATURE_PATHS))
    feature_path = source / feature_rel
    raw_features = _read_bytes(feature_path)
    checksums[str(feature_rel)] = _sha256(raw_features)
    try:
        feature_data = json.loads(raw_features.decode("utf-8"))
        feature_data, redacted_paths = _sanitize_record_paths(feature_data, source)
        work_items = parse_feature_records(feature_data, origin=str(feature_rel))
    except (UnicodeDecodeError, ValueError) as error:
        raise MigrationError(f"Relay features at {feature_rel} are not valid JSON: {error}") from error
    if redacted_paths:
        warnings.append(f"absolute host paths were redacted in {feature_rel}")
    if feature_rel == Path("features.json"):
        warnings.append("Relay features were read from the bare features.json fallback path")

    progress_rel = _find_first(source, PROGRESS_PATHS)
    if progress_rel is None:
        warnings.append("no Relay progress file found; looked for " + ", ".join(str(p) for p in PROGRESS_PATHS))
        entries: list[dict[str, Any]] = []
    else:
        raw_progress_bytes = _read_bytes(source / progress_rel)
        checksums[str(progress_rel)] = _sha256(raw_progress_bytes)
        try:
            raw_progress, redacted_paths = _sanitize_record_paths(raw_progress_bytes.decode("utf-8"), source)
            entries = parse_progress_text(raw_progress, source_file=str(progress_rel))
        except UnicodeDecodeError as error:
            raise MigrationError(f"Relay progress at {progress_rel} is not UTF-8: {error}") from error
        if progress_rel == Path("claude-progress.txt"):
            warnings.append("Relay progress was read from the legacy claude-progress.txt")
        if redacted_paths:
            warnings.append(f"absolute host paths were redacted in {progress_rel}")

    spec_rel = _find_first(source, SPEC_PATHS)
    if spec_rel is None:
        warnings.append("no Relay project specification found; looked for " + ", ".join(str(p) for p in SPEC_PATHS))
        spec_text = ""
    else:
        spec_bytes = _read_bytes(source / spec_rel)
        spec_text, redacted_paths = _sanitize_record_paths(spec_bytes.decode("utf-8", errors="replace"), source)
        checksums[str(spec_rel)] = _sha256(spec_bytes)
        if redacted_paths:
            warnings.append(f"absolute host paths were redacted in {spec_rel}")

    outcomes: list[dict[str, Any]] = []
    for candidate in OUTCOME_DIRS:
        directory = source / candidate
        if not directory.is_dir():
            continue
        kind = "current" if candidate == OUTCOME_DIRS[0] else "legacy"
        if kind == "legacy" and outcomes:
            continue
        records = sorted(directory.glob("*.json"))
        if kind == "legacy" and not records:
            continue
        for record_path in records:
            relative = str(record_path.relative_to(source))
            raw_outcome = _read_bytes(record_path)
            checksums[relative] = _sha256(raw_outcome)
            try:
                raw_record = json.loads(raw_outcome.decode("utf-8"))
                safe_record, redacted_paths = _sanitize_record_paths(raw_record, source)
                outcomes.append(
                    parse_outcome_record(
                        safe_record,
                        source_path=relative,
                        source_kind=kind,
                    )
                )
                if redacted_paths:
                    warnings.append(f"absolute host paths were redacted in {relative}")
            except (UnicodeDecodeError, ValueError) as error:
                raise MigrationError(f"session outcome {relative} is not valid JSON: {error}") from error
        if kind == "legacy":
            warnings.append("session outcomes were read from the legacy .agent_harness directory")
    if not outcomes:
        warnings.append("no Relay session outcomes were found")

    return work_items, entries, outcomes, spec_text, raw_progress if progress_rel else "", checksums, warnings


def migration_identity(source_identity: str, source_head: str, checksums: dict[str, str]) -> str:
    seed = json.dumps([source_identity, source_head, sorted(checksums.items())], separators=(",", ":"))
    return "relay-" + _sha256(seed.encode("utf-8"))[:16]


def build_manifest(
    source: Path,
    *,
    created_at: str | None = None,
    harness_version: str | None = None,
    max_commits: int = 20,
) -> RelayMigrationManifest:
    """Snapshot a Relay repository into a manifest without writing to it."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise MigrationError(f"Relay source {source} is not a directory")

    work_items, entries, outcomes, spec_text, raw_progress, checksums, warnings = read_source(source)
    created = created_at or _utc_now()
    head = _git(source, "rev-parse", "HEAD") or ""
    if not head:
        raise MigrationError("Relay source must be a Git repository with a resolvable HEAD")
    source_status = _git(source, "status", "--porcelain") or ""
    dirty = bool(source_status)
    if dirty:
        warnings.append("Relay source worktree is dirty; the manifest describes HEAD plus local changes")

    version = harness_version or _read_harness_version(source)
    if version == "unknown":
        warnings.append("Relay source harness version is unknown")

    commits_by_sha: dict[str, dict[str, Any]] = {}
    for input_path in sorted(checksums):
        log = _git(
            source,
            "log",
            f"-{max(1, max_commits)}",
            "--format=%H%x1f%ct%x1f%s",
            "--",
            input_path,
        )
        if not log:
            continue
        sha, _, rest = log.splitlines()[0].partition("\x1f")
        timestamp, _, message = rest.partition("\x1f")
        if _SHA_RE.match(sha.strip()):
            commit = commits_by_sha.setdefault(
                sha.strip(),
                {
                    "sha": sha.strip(),
                    "message": _sanitize_record_paths(message.strip(), source)[0],
                    "timestamp": timestamp.strip(),
                    "files": [],
                },
            )
            commit["files"].append(input_path)
    commits = list(commits_by_sha.values())
    commit_for_file = {file: commit for commit in commits for file in commit["files"]}
    for item in work_items:
        commit = commit_for_file.get(item["source_file"])
        if commit:
            item["source_commit"] = commit["sha"]
            item["source_timestamp"] = item["last_updated"] or commit["timestamp"]
    for entry in entries:
        commit = commit_for_file.get(entry["source_file"])
        if commit:
            entry["source_commit"] = commit["sha"]
            entry["source_timestamp"] = commit["timestamp"]
    for outcome in outcomes:
        commit = commit_for_file.get(outcome["source_path"])
        if commit:
            outcome["source_commit"] = commit["sha"]
            outcome["source_timestamp"] = str(
                outcome["source_record"].get("timestamp")
                or outcome["source_record"].get("created_at")
                or commit["timestamp"]
            )

    identity = sanitize_repository_identity(source)
    migration_id = migration_identity(identity, head, checksums)
    provenance = {
        "migration_id": migration_id,
        "source_repository": identity,
        "source_head": head,
        "source_kind": EVIDENCE_KIND,
    }
    return RelayMigrationManifest(
        schema_version=SCHEMA_VERSION,
        migration_id=migration_id,
        source_repository=provenance["source_repository"],
        source_head=head,
        source_harness_version=version,
        source_dirty=dirty,
        project_spec=spec_text,
        work_items=work_items,
        progress_entries=entries,
        session_outcomes=outcomes,
        source_commits=commits,
        provenance=provenance,
        checksums=checksums,
        warnings=warnings,
        created_at=created,
        raw_progress=raw_progress,
        input_files=sorted(checksums),
        source_status=source_status,
    )


def _read_harness_version(source: Path) -> str:
    for name in ("VERSION", "version.txt"):
        candidate = source / name
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    return "unknown"


def detect_drift(manifest: RelayMigrationManifest, source: Path) -> list[str]:
    """What changed in the source since the manifest was built."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise MigrationError(f"Relay source {source} is not a directory")
    drift: list[str] = []
    head = _git(source, "rev-parse", "HEAD") or ""
    if manifest.source_head and head != manifest.source_head:
        drift.append(f"source HEAD moved from {manifest.source_head[:12]} to {(head or 'none')[:12]}")
    current_status = _git(source, "status", "--porcelain") or ""
    if manifest.source_status != current_status:
        old_status = set(manifest.source_status.splitlines())
        new_status = set(current_status.splitlines())
        for line in sorted(old_status ^ new_status):
            relative = line[3:].strip() if len(line) > 3 else line.strip()
            drift.append(f"source worktree changed at {relative}")
    for relative, expected in sorted(manifest.checksums.items()):
        path = source / relative
        if not path.is_file():
            drift.append(f"{relative} was removed after the manifest was built")
            continue
        if _sha256(_read_bytes(path)) != expected:
            drift.append(f"{relative} changed after the manifest was built")
    try:
        *_, current_checksums, _warnings = read_source(source)
    except MigrationError as error:
        drift.append(f"source inputs can no longer be read: {error}")
    else:
        for relative in sorted(set(current_checksums) - set(manifest.checksums)):
            drift.append(f"new source input {relative} appeared after the manifest was built")
        for relative in sorted(set(manifest.checksums) - set(current_checksums)):
            if not (source / relative).exists():
                continue
            drift.append(f"source input set changed for {relative}")
    return drift


def assert_fresh(manifest: RelayMigrationManifest, source: Path) -> None:
    drift = detect_drift(manifest, source)
    if drift:
        raise StaleManifestError("refusing a stale manifest; regenerate it and retry. Changed: " + "; ".join(drift))


def validate_manifest_source(manifest: RelayMigrationManifest, source: Path) -> None:
    """Bind every importable manifest field to a fresh read of its source."""
    assert_fresh(manifest, source)
    current = build_manifest(source)
    captured = manifest.to_dict()
    rebuilt = current.to_dict()
    captured.pop("created_at", None)
    rebuilt.pop("created_at", None)
    if captured != rebuilt:
        raise StaleManifestError(
            "manifest payload or provenance does not match the checksummed Relay source; recapture the manifest"
        )


def _same_repository(source: Path, target: Path) -> bool:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if source == target:
        return True
    source_git = _git(source, "rev-parse", "--path-format=absolute", "--git-common-dir")
    target_git = _git(target, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return bool(source_git and target_git and Path(source_git).resolve() == Path(target_git).resolve())


# --------------------------------------------------------------------------
# Raven target
# --------------------------------------------------------------------------

try:  # Raven owns the on-disk shape; use its own writers when importable.
    from raven.stint import backlog as _backlog
    from raven.stint import journal as _journal
except Exception:  # pragma: no cover - exercised only outside a Raven checkout
    _backlog = None
    _journal = None


def _backlog_module():
    if _backlog is None:
        raise MigrationError("raven.stint.backlog is not importable; run this from a Raven checkout")
    return _backlog


def _journal_module():
    if _journal is None:
        raise MigrationError("raven.stint.journal is not importable; run this from a Raven checkout")
    return _journal


def _assert_not_proven(task: Any) -> None:
    """No path through this importer may leave a task looking proven."""
    module = _backlog_module()
    if task.state in (module.DONE, module.IN_REVIEW):
        raise MigrationError(f"refusing to write task {task.id} in state {task.state!r}")
    if "accept" in str(getattr(task, "implements", "")).lower():
        raise MigrationError(f"refusing to write acceptance evidence on task {task.id}")


def _marker(provenance: dict[str, str], kind: str, key: str) -> str:
    return f"relay-migration:{provenance['migration_id']}:{kind}:{key}"


def map_work_item(item: dict[str, Any], provenance: dict[str, str]) -> dict[str, Any]:
    """Decide the Raven state a Relay item may legitimately arrive in.

    A Relay completion is a claim, not a verdict, so it lands blocked on a
    human review rather than done. Anything Relay could not finish stays open
    or blocked and keeps its source status in the note, so the failure is
    still visible in the board.
    """
    module = _backlog_module()
    claimed = item["claimed_complete"]
    failed = item["relay_status"] == "failed"
    state = module.BLOCKED if (claimed or failed or item["relay_status"] == "blocked") else module.OPEN

    reasons: list[str] = []
    if item["passes_claim"]:
        reasons.append("Relay claimed passes=true")
    if item["relay_status"] == "completed" and not item["passes_claim"]:
        reasons.append(f"Relay status was {item['source_status']}")
    if failed:
        reasons.append("Relay status was failed")
    if item["relay_status"] == "blocked":
        reasons.append("Relay status was blocked")

    note_lines = [
        f"Relay feature {item['feature_id']}: source status {item['source_status']}.",
        f"Provenance: {_provenance_line(provenance)}.",
        f"Relay source file: {item['source_file'] or 'unknown'}.",
        "Imported as a legacy claim only: not proven, not verified, not accepted.",
    ]
    if item["source_commit"]:
        note_lines.append(f"Relay source commit: {item['source_commit']} ({item['source_timestamp']}).")
    if reasons:
        note_lines.append("Why it is not done: " + "; ".join(reasons) + ".")
    if item["notes"]:
        note_lines.append(f"Relay notes: {item['notes']}")
    if item["last_updated"]:
        note_lines.append(f"Relay last_updated: {item['last_updated']}")
    if item["steps"]:
        note_lines.append("Relay acceptance criteria: " + " | ".join(item["steps"]))

    title = item["title"] if len(item["title"]) <= 96 else item["title"][:95].rstrip() + "…"
    task = module.Task(
        id=0,
        title=f"{item['feature_id']}: {title}",
        name=item["feature_id"][:48],
        source=provenance["source_repository"],
        state=state,
        blocked_by=[UNVERIFIED_BLOCKER] if claimed else [],
        note="\n".join(note_lines),
    )
    task.history = [
        {
            "from": "",
            "to": state,
            "actor": "planner",
            "via": "relay-migration",
            "at": _provenance_line(provenance),
            "note": _marker(provenance, "feature", item["feature_id"]),
        }
    ]
    _assert_not_proven(task)
    return {"task": task, "state": state, "marker": _marker(provenance, "feature", item["feature_id"])}


def _new_plan(migration_id: str, target: str) -> dict[str, Any]:
    return {
        "migration_id": migration_id,
        "target": target,
        "new_tasks": [],
        "skipped_tasks": [],
        "progress_entries": [],
        "outcomes": [],
        "failures": [],
        "writes": [],
    }


def plan_import(manifest: RelayMigrationManifest, target: Path, *, source: Path | None = None) -> dict[str, Any]:
    """Build the import plan, reading the target but never writing to it."""
    target = Path(target).expanduser().resolve()
    if source is not None and _same_repository(source, target):
        raise MigrationError("source and target must be different repositories")
    module = _backlog_module()
    existing = module.Backlog()
    if module.backlog_path(target).is_file():
        existing = module.load(target)

    markers = {
        str(entry.get("reason", entry.get("note", "")))
        for task in existing.tasks
        for entry in (task.history or ())
        if str(entry.get("reason", entry.get("note", ""))).startswith("relay-migration:")
    }

    plan = _new_plan(manifest.migration_id, str(target))
    for item in manifest.work_items:
        planned = map_work_item(item, manifest.provenance)
        if planned["marker"] in markers:
            plan["skipped_tasks"].append(item["feature_id"])
        else:
            plan["new_tasks"].append(planned)
    plan["progress_entries"] = list(manifest.progress_entries)
    plan["outcomes"] = list(manifest.session_outcomes)
    plan["failures"] = [outcome for outcome in plan["outcomes"] if _is_failure(outcome)]
    receipt_root = RECEIPT_DIR / manifest.migration_id
    plan["writes"] = [
        str(module.backlog_path(target)),
        str(target / _journal_module().JOURNAL),
        str(target / receipt_root / RECEIPT_FILE),
        str(target / receipt_root / "manifest.json"),
    ]
    return plan


def _render_journal_section(manifest: RelayMigrationManifest, plan: dict[str, Any]) -> str:
    """The journal entry: an ordered account that points at the receipt.

    The journal clips one entry, so the full text and outcome records live in
    the receipt and manifest; the journal keeps the order and the summary.
    """
    lines = [
        f"Imported Relay history ({_provenance_line(manifest.provenance)}).",
        "",
        f"- work items: {len(manifest.work_items)} "
        f"({sum(1 for item in manifest.work_items if item['claimed_complete'])} claimed complete by Relay, "
        "all imported as unproven and review-blocked)",
        f"- progress entries: {len(manifest.progress_entries)} (order and text preserved in the receipt)",
        f"- session outcomes: {len(manifest.session_outcomes)} "
        f"({len(plan['failures'])} failures preserved), all labelled {EVIDENCE_KIND}",
        f"- source commits: {len(manifest.source_commits)}",
        f"- full record: {(RECEIPT_DIR / manifest.migration_id / RECEIPT_FILE).as_posix()}",
        "",
        "Nothing here is AO acceptance evidence and nothing is proven.",
    ]
    if manifest.warnings:
        lines += ["", "Warnings:"] + [f"- {warning}" for warning in manifest.warnings]
    return "\n".join(lines)


def _render_receipt(manifest: RelayMigrationManifest, plan: dict[str, Any]) -> dict[str, Any]:
    """The durable receipt: everything the journal clips, kept verbatim."""
    return {
        "schema": "raven.relay-migration-receipt",
        "schema_version": SCHEMA_VERSION,
        "migration_id": manifest.migration_id,
        "imported_at": _utc_now(),
        "provenance": manifest.provenance,
        "source_head": manifest.source_head,
        "source_repository": manifest.source_repository,
        "source_harness_version": manifest.source_harness_version,
        "evidence_kind": EVIDENCE_KIND,
        "acceptance_evidence": False,
        "project_spec_snapshot": manifest.project_spec,
        "raw_progress": manifest.raw_progress,
        "progress_entries": [entry for entry in manifest.progress_entries],
        "session_outcomes": [outcome for outcome in manifest.session_outcomes],
        "source_commits": [commit for commit in manifest.source_commits],
        "warnings": list(manifest.warnings),
        "checksums": dict(manifest.checksums),
        "imported_features": [
            {
                "feature_id": item["feature_id"],
                "relay_status": item["relay_status"],
                "source_status": item["source_status"],
                "imported_state": map_work_item(item, manifest.provenance)["state"],
                "marker": _marker(manifest.provenance, "feature", item["feature_id"]),
                "source_file": item["source_file"],
                "source_commit": item["source_commit"],
                "source_timestamp": item["source_timestamp"],
                "source_record": item["source_record"],
            }
            for item in manifest.work_items
        ],
        "skipped_existing": list(plan["skipped_tasks"]),
    }


def apply_import(
    manifest: RelayMigrationManifest,
    target: Path,
    *,
    apply: bool,
    source: Path | None = None,
) -> dict[str, Any]:
    """Plan, then write -- only when `apply` is true and a target is given.

    Writes are per-file and atomic, and every write is a merge keyed on the
    migration marker, so an interrupted import resumes by rerunning rather than
    needing a transaction.
    """
    if not apply:
        return plan_import(manifest, target)
    if target is None:
        raise MigrationError("applying needs an explicit target project")
    if source is None:
        raise MigrationError("applying needs the explicit Relay source repository")

    target = Path(target).expanduser().resolve()
    validate_manifest_source(manifest, source)
    if _same_repository(source, target):
        raise MigrationError("source and target must be different repositories")
    module = _backlog_module()
    plan = plan_import(manifest, target, source=source)

    if plan["new_tasks"]:
        for planned in plan["new_tasks"]:
            task = planned["task"]
            backlog = module.start(target)
            task = module.apply(
                backlog,
                "add",
                role=module.BOOTSTRAP,
                title=task.title,
                name=task.name,
                source=task.source,
                note=task.note,
                reason=planned["marker"],
            )
            if planned["state"] == module.BLOCKED:
                module.apply(
                    backlog,
                    "block",
                    role=module.PLANNER,
                    task_id=task.id,
                    blocker=UNVERIFIED_BLOCKER,
                    reason=task.note,
                )
            task.history[-1]["via"] = "relay-migration"
            task.history[-1]["migration_id"] = manifest.migration_id
            _assert_not_proven(task)
            module.save(target, backlog)

    if manifest.progress_entries or manifest.warnings:
        journal = _journal_module()
        path = target / journal.JOURNAL
        entry = _render_journal_section(manifest, plan)
        marker = f"Relay migration {manifest.migration_id}"
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        section = f"## {marker}\n\n{entry}\n"
        if section not in current:
            heading = current.rfind("## " + marker)
            if heading >= 0:
                next_heading = current.find("\n## ", heading + len("## " + marker))
                prefix = current[:heading].rstrip("\n")
                suffix = current[next_heading + 1 :] if next_heading >= 0 else ""
                current = prefix + "\n\n" + section + ("\n" + suffix if suffix else "")
            elif current:
                current = current.rstrip("\n") + "\n\n" + section
            else:
                current = "# Journal\n\n" + section
            _atomic_write(path, current)

    if manifest.project_spec.strip():
        spec_path = target / RECEIPT_DIR / manifest.migration_id / LEGACY_SPEC_FILE
        header = (
            "# Legacy Relay specification\n\n"
            f"Snapshot of the Relay project's own specification: {_provenance_line(manifest.provenance)}.\n"
            "This is not Raven's SPEC.md and is not a Raven plan.\n\n---\n\n"
        )
        if not spec_path.is_file() or spec_path.read_text(encoding="utf-8") != header + manifest.project_spec:
            _atomic_write(spec_path, header + manifest.project_spec)

    receipt_path = target / RECEIPT_DIR / manifest.migration_id / RECEIPT_FILE
    manifest_path = target / RECEIPT_DIR / manifest.migration_id / "manifest.json"
    _atomic_write(manifest_path, manifest.dump())
    receipt = _render_receipt(manifest, plan)
    existing: dict[str, Any] = {}
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    if existing.get("migration_id") == manifest.migration_id and existing.get("imported_at"):
        receipt["imported_at"] = str(existing["imported_at"])
        receipt["import_runs"] = int(existing.get("import_runs", 1)) + 1
    else:
        receipt["import_runs"] = 1
    _atomic_write(receipt_path, json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return plan


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_report(manifest: RelayMigrationManifest, plan: dict[str, Any]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for item in manifest.work_items:
        by_status[item["relay_status"]] = by_status.get(item["relay_status"], 0) + 1
    outcome_status: dict[str, int] = {}
    for outcome in manifest.session_outcomes:
        outcome_status[outcome["status"]] = outcome_status.get(outcome["status"], 0) + 1
    by_state: dict[str, int] = {}
    for planned in plan["new_tasks"]:
        by_state[planned["state"]] = by_state.get(planned["state"], 0) + 1
    return {
        "schema_version": manifest.schema_version,
        "migration_id": manifest.migration_id,
        "source_repository": manifest.source_repository,
        "source_head": manifest.source_head,
        "source_harness_version": manifest.source_harness_version,
        "evidence_kind": EVIDENCE_KIND,
        "project_spec_present": bool(manifest.project_spec.strip()),
        "work_items_total": len(manifest.work_items),
        "work_items_by_status": dict(sorted(by_status.items())),
        "work_items_claimed_complete": sum(1 for item in manifest.work_items if item["claimed_complete"]),
        "planned_tasks_by_state": dict(sorted(by_state.items())),
        "already_imported_features": list(plan["skipped_tasks"]),
        "progress_entries": len(manifest.progress_entries),
        "session_outcomes_total": len(manifest.session_outcomes),
        "session_outcomes_by_status": dict(sorted(outcome_status.items())),
        "failed_session_outcomes": [outcome["session_id"] for outcome in plan["failures"]],
        "source_commits": len(manifest.source_commits),
        "checksummed_inputs": len(manifest.checksums),
        "warnings": list(manifest.warnings),
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Relay migration {report['migration_id']}",
        f"  source: {report['source_repository']} @ {(report['source_head'] or 'unknown')[:12]} "
        f"(harness {report['source_harness_version']})",
        f"  project spec snapshot: {'yes' if report['project_spec_present'] else 'no'}",
        f"  work items: {report['work_items_total']} by status {report['work_items_by_status']}",
        f"  claimed complete by Relay: {report['work_items_claimed_complete']} "
        "(imported unproven and review-blocked, never done)",
        f"  planned Raven states: {report['planned_tasks_by_state']}",
        f"  already imported (merge-on-rerun): {len(report['already_imported_features'])}",
        f"  progress entries: {report['progress_entries']}",
        f"  session outcomes: {report['session_outcomes_total']} by status {report['session_outcomes_by_status']}",
        f"  failed outcomes preserved: {report['failed_session_outcomes']}",
        f"  source commits: {report['source_commits']}",
        f"  checksummed inputs: {report['checksummed_inputs']}",
    ]
    for warning in report["warnings"]:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relay-migration",
        description="Import a read-only Relay repository into a Raven stint project.",
    )
    parser.add_argument("command", choices=("capture", "import"), nargs="?", default="capture")
    parser.add_argument("--source", required=True, type=Path, help="Relay repository to read (never written)")
    parser.add_argument("--target", type=Path, default=None, help="Raven project to write")
    parser.add_argument("--manifest", type=Path, default=None, help="Saved manifest for import")
    parser.add_argument("--apply", action="store_true", help="Actually write; without it the run is a dry run")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Write the captured manifest JSON here")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "capture":
            manifest = build_manifest(args.source)
            if args.manifest_out:
                _atomic_write(Path(args.manifest_out), manifest.dump())
        else:
            if args.manifest is None:
                raise MigrationError("import needs --manifest captured by a prior run")
            manifest = RelayMigrationManifest.load(args.manifest)
            if sanitize_repository_identity(args.source) != manifest.source_repository:
                raise MigrationError("--source does not match the manifest source repository")
            validate_manifest_source(manifest, args.source)
        if args.apply:
            if args.target is None:
                raise MigrationError("--apply needs --target: a dry run writes nothing")
            plan = apply_import(manifest, args.target, apply=True, source=args.source)
        elif args.target is None and args.command == "capture":
            plan = _new_plan(manifest.migration_id, "(no target selected)")
        elif args.target is None:
            raise MigrationError("import needs --target; without --apply it is a dry run")
        else:
            plan = plan_import(manifest, args.target, source=args.source)
    except MigrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = build_report(manifest, plan)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
