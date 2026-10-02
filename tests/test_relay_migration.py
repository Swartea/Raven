"""Coverage for the Relay -> Raven migration.

The fixture is a real git repository built in a temp directory, because half of
what makes this migration safe is only true of real repositories: a real HEAD to
drift away from and a real worktree to leave untouched.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("relay_migration", REPO_ROOT / "scripts" / "relay_migration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["relay_migration"] = module
    spec.loader.exec_module(module)
    return module


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

rm = None


rm = _load_module()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _tree_snapshot(root: Path) -> dict[Path, bytes | None]:
    return {path.relative_to(root): None if path.is_dir() else path.read_bytes() for path in root.rglob("*")}


PROGRESS = """## Session 1 - 2026-01-04
Implemented the retry path.
Smoke test passed.

## Session 2 - 2026-01-05
Ran the reviewer and fixed two findings.
"""


FEATURES = [
    {"id": "alpha", "title": "Alpha path", "status": "pending", "steps": ["alpha works"]},
    {"id": "beta", "title": "Beta path", "passes": True, "notes": "claimed green", "last_updated": "2026-01-05"},
    {"id": "gamma", "title": "Gamma path", "status": "completed", "steps": ["gamma works"]},
    {"id": "delta", "title": "Delta path", "status": "failed", "notes": "crashed on input"},
    {"id": "epsilon", "title": "Epsilon path", "status": "blocked", "notes": "waiting on API key"},
    {"id": "zeta", "title": "Zeta path", "status": "in_progress", "notes": "half done", "steps": ["z1", "z2"]},
]

OUTCOMES = [
    {
        "session_id": "sess-1",
        "feature_id": "alpha",
        "status": "complete",
        "message": "done",
        "smoke_test_passed": True,
        "commit_created": True,
        "verification_passed": True,
    },
    {
        "session_id": "sess-2",
        "feature_id": "delta",
        "status": "verification_failed",
        "failure_reason": "reviewer rejected the patch",
        "verification_passed": False,
        "qa_passed": False,
        "qa_failure_stage": "qa",
        "qa_results": [{"check": "lint", "passed": False}],
    },
    {
        "session_id": "sess-3",
        "feature_id": "epsilon",
        "status": "needs_review",
        "message": "blocked on credentials",
    },
]


@pytest.fixture()
def relay_repo(tmp_path: Path) -> Path:
    """A real git repository holding a realistic Relay history."""
    repo = tmp_path / "relay-source"
    (repo / ".agent_relay" / "session_outcomes").mkdir(parents=True)
    (repo / "src").mkdir()

    (repo / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "relay@example.com")
    _git(repo, "config", "user.name", "Relay Harness")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feat: add the relay entry point")

    _write_json(repo / "feature_list.json", {"features": FEATURES})
    (repo / "agent-progress.txt").write_text(PROGRESS, encoding="utf-8")
    (repo / "project_spec.md").write_text("# Relay project\n\nShip the retry path.\n", encoding="utf-8")
    for outcome in OUTCOMES:
        _write_json(repo / ".agent_relay" / "session_outcomes" / f"{outcome['session_id']}.json", outcome)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: record relay execution history")
    return repo


@pytest.fixture()
def target_project(tmp_path: Path) -> Path:
    target = tmp_path / "raven-project"
    target.mkdir()
    (target / ".stint").mkdir()
    rm._backlog_module().save(target, rm._backlog_module().Backlog())
    return target


# -- parser compatibility -------------------------------------------------


def test_parses_legacy_field_spellings():
    items = rm.parse_feature_records(
        [
            {"id": "a", "passes": True, "steps": "one\ntwo"},
            {"id": "b", "acceptance_criteria": ["x"]},
            {"id": "c", "status": "nonsense"},
            {"id": "d"},
        ],
        origin="features.json",
    )
    assert items[0]["claimed_complete"] is True and items[0]["passes_claim"] is True
    assert items[0]["steps"] == ["one", "two"]
    assert items[1]["steps"] == ["x"]
    assert items[2]["relay_status"] == "pending" and items[3]["relay_status"] == "pending"


def test_passes_outranks_completed_and_rejects_malformed():
    items = rm.parse_feature_records([{"id": "a", "status": "completed", "passes": False}], origin="f.json")
    assert items[0]["claimed_complete"] is True
    with pytest.raises(rm.MigrationError):
        rm.parse_feature_records([{"no": "id"}], origin="f.json")
    with pytest.raises(rm.MigrationError):
        rm.parse_feature_records(["not-a-record"], origin="f.json")


def test_progress_keeps_order_and_raw_text():
    entries = rm.parse_progress_text(PROGRESS)
    assert [entry["index"] for entry in entries] == list(range(len(entries)))
    assert "Implemented the retry path." in entries[0]["raw"]
    assert "## Session 2 - 2026-01-05" in entries[1]["raw"]
    crlf = rm.parse_progress_text(PROGRESS.replace("\n", "\r\n"))
    assert "\r\n" in crlf[0]["raw"]


def test_outcome_record_keeps_booleans_and_no_invented_sha():
    outcome = rm.parse_outcome_record(OUTCOMES[0], source_path="x.json", source_kind="current")
    assert outcome["commit_created"] is True
    assert not hasattr(outcome, "commit_sha")
    assert outcome["evidence_kind"] == rm.EVIDENCE_KIND


# -- manifest, checksums, provenance --------------------------------------


def test_manifest_is_versioned_and_checksummed(relay_repo):
    manifest = rm.build_manifest(relay_repo)
    assert manifest.schema_version == rm.SCHEMA_VERSION
    assert manifest.migration_id.startswith("relay-")
    assert manifest.source_head == _git(relay_repo, "rev-parse", "HEAD")
    assert manifest.source_harness_version == "unknown"
    assert any("harness version" in w for w in manifest.warnings)
    assert "feature_list.json" in manifest.checksums
    assert [c["sha"] for c in manifest.source_commits]
    assert all(c["message"] for c in manifest.source_commits)
    assert rm.RelayMigrationManifest.from_dict(manifest.to_dict()).migration_id == manifest.migration_id
    later_capture = rm.build_manifest(relay_repo, created_at="2030-01-01T00:00:00+00:00")
    assert later_capture.migration_id == manifest.migration_id


def test_manifest_rejects_a_foreign_schema(relay_repo):
    manifest = rm.build_manifest(relay_repo)
    payload = manifest.to_dict()
    payload["schema_version"] = 99
    with pytest.raises(rm.MigrationError):
        rm.RelayMigrationManifest.from_dict(payload)


def test_source_identity_is_not_a_host_path(relay_repo):
    manifest = rm.build_manifest(relay_repo)
    assert "/" not in manifest.source_repository and "\\" not in manifest.source_repository


def test_legacy_paths_are_read_with_a_warning(relay_repo, tmp_path):
    legacy = tmp_path / "legacy-source"
    (legacy / ".agent_harness" / "session_outcomes").mkdir(parents=True)
    (legacy / "src").mkdir()
    (legacy / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(legacy))
    _git(legacy, "config", "user.email", "l@example.com")
    _git(legacy, "config", "user.name", "L")
    _write_json(legacy / ".agent_harness" / "features.json", [{"id": "a", "passes": True}])
    (legacy / "claude-progress.txt").write_text("## Session 1\nlegacy\n", encoding="utf-8")
    _write_json(legacy / ".agent_harness" / "session_outcomes" / "s1.json", OUTCOMES[0])
    _git(legacy, "add", "-A")
    _git(legacy, "commit", "-q", "-m", "feat: legacy relay state")

    manifest = rm.build_manifest(legacy)
    assert manifest.session_outcomes[0]["source_kind"] == "legacy"
    assert manifest.session_outcomes[0]["evidence_kind"] == rm.EVIDENCE_KIND
    assert any("claude-progress" in w for w in manifest.warnings)
    assert any(".agent_harness" in w for w in manifest.warnings)


def test_missing_optional_inputs_warn_but_malformed_required_raises(tmp_path):
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    (minimal / "src").mkdir()
    (minimal / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(minimal))
    _git(minimal, "config", "user.email", "m@example.com")
    _git(minimal, "config", "user.name", "M")
    _write_json(minimal / ".agent_relay" / "features.json", [{"id": "a"}])
    _git(minimal, "add", "-A")
    _git(minimal, "commit", "-q", "-m", "feat: minimal relay")

    manifest = rm.build_manifest(minimal)
    assert manifest.project_spec == ""
    assert any("progress" in w for w in manifest.warnings)
    assert any("specification" in w for w in manifest.warnings)
    assert any("session outcomes" in w for w in manifest.warnings)

    (minimal / ".agent_relay" / "features.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(rm.MigrationError):
        rm.build_manifest(minimal)


# -- dry run, mapping, provenance -----------------------------------------


def test_dry_run_writes_nothing(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    before = _tree_snapshot(target_project)
    plan = rm.plan_import(manifest, target_project)
    report = rm.build_report(manifest, plan)
    assert report["work_items_total"] == 6
    assert report["work_items_by_status"]["pending"] == 2
    assert report["progress_entries"] == 2
    assert report["source_commits"] == 1
    assert "feature_list.json" in manifest.source_commits[0]["files"]
    assert _tree_snapshot(target_project) == before
    assert rm.main(["--source", str(relay_repo), "--target", str(target_project)]) == 0


def test_capture_then_import_uses_saved_manifest_and_dry_run_is_read_only(relay_repo, target_project, tmp_path):
    manifest_path = tmp_path / "captured-manifest.json"
    before = _tree_snapshot(target_project)

    assert rm.main(["capture", "--source", str(relay_repo), "--manifest-out", str(manifest_path), "--json"]) == 0
    assert manifest_path.is_file()
    assert (
        rm.main(
            [
                "import",
                "--source",
                str(relay_repo),
                "--manifest",
                str(manifest_path),
                "--target",
                str(target_project),
                "--json",
            ]
        )
        == 0
    )
    assert _tree_snapshot(target_project) == before

    assert (
        rm.main(
            [
                "import",
                "--source",
                str(relay_repo),
                "--manifest",
                str(manifest_path),
                "--target",
                str(target_project),
                "--apply",
            ]
        )
        == 0
    )
    assert len(rm._backlog_module().load(target_project).tasks) == 6


def test_status_mapping_never_proves_legacy_work(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    tasks = {t.name: t for t in rm._backlog_module().load(target_project).tasks}
    module = rm._backlog_module()

    assert tasks["alpha"].state == module.OPEN
    assert tasks["zeta"].state == module.OPEN
    for name in ("beta", "gamma"):
        assert tasks[name].state == module.BLOCKED
        assert rm.UNVERIFIED_BLOCKER in tasks[name].blocked_by
        assert "not proven" in tasks[name].note
    assert tasks["delta"].state == module.BLOCKED and "failed" in tasks["delta"].note
    assert tasks["epsilon"].state == module.BLOCKED

    for task in tasks.values():
        assert task.state != module.DONE
        assert task.state != module.IN_REVIEW
        assert not task.implements


def test_provenance_and_order_are_preserved(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    tasks = rm._backlog_module().load(target_project).tasks
    for task in tasks:
        assert task.history and any(entry.get("via") == "relay-migration" for entry in task.history)
        assert any(manifest.migration_id in entry.get("reason", "") for entry in task.history)
        assert task.source == manifest.source_repository

    receipt = json.loads(
        (target_project / rm.RECEIPT_DIR / manifest.migration_id / rm.RECEIPT_FILE).read_text(encoding="utf-8")
    )
    assert [e["raw"] for e in receipt["progress_entries"]] == [e["raw"] for e in manifest.progress_entries]
    assert receipt["acceptance_evidence"] is False
    assert receipt["evidence_kind"] == rm.EVIDENCE_KIND
    assert "Ship the retry path." in receipt["project_spec_snapshot"]
    assert receipt["raw_progress"] == PROGRESS
    assert not (target_project / "SPEC.md").exists()

    journal = (target_project / "JOURNAL.md").read_text(encoding="utf-8")
    assert "Relay migration" in journal and "Nothing here is AO acceptance evidence" in journal


def test_failed_outcomes_survive_the_import(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    report = rm.build_report(manifest, rm.plan_import(manifest, target_project))
    assert sorted(report["failed_session_outcomes"]) == ["sess-2"]
    assert report["session_outcomes_by_status"]["needs_review"] == 1
    receipt = json.loads(
        (target_project / rm.RECEIPT_DIR / manifest.migration_id / rm.RECEIPT_FILE).read_text(encoding="utf-8")
    )
    failed = [o for o in receipt["session_outcomes"] if o["session_id"] == "sess-2"][0]
    assert failed["failure_reason"] == "reviewer rejected the patch"
    assert failed["qa_failure_stage"] == "qa" and failed["verification_passed"] is False


# -- idempotency, drift, recovery -----------------------------------------


def test_repeat_import_is_idempotent(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    first = rm._backlog_module().load(target_project).to_dict()
    journal_first = (target_project / "JOURNAL.md").read_text(encoding="utf-8")

    again = rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    assert again["new_tasks"] == []
    assert len(again["skipped_tasks"]) == 6
    assert rm._backlog_module().load(target_project).to_dict() == first
    assert (target_project / "JOURNAL.md").read_text(encoding="utf-8").count(
        f"## Relay migration {manifest.migration_id}"
    ) == 1
    assert journal_first.count("Imported Relay history") == 1
    receipt = json.loads(
        (target_project / rm.RECEIPT_DIR / manifest.migration_id / rm.RECEIPT_FILE).read_text(encoding="utf-8")
    )
    assert receipt["import_runs"] == 2


def test_source_drift_is_rejected_with_a_regenerate_hint(relay_repo):
    manifest = rm.build_manifest(relay_repo)
    (relay_repo / "src" / "app.py").write_text("print('drift')\n", encoding="utf-8")
    drift = rm.detect_drift(manifest, relay_repo)
    assert any("app.py" in d for d in drift)
    with pytest.raises(rm.StaleManifestError) as excinfo:
        rm.assert_fresh(manifest, relay_repo)
    assert "regenerate" in str(excinfo.value)

    (relay_repo / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (relay_repo / "src" / "new.py").write_text("y = 2\n", encoding="utf-8")
    _git(relay_repo, "add", "-A")
    _git(relay_repo, "commit", "-q", "-m", "chore: move HEAD")
    assert any("HEAD moved" in d for d in rm.detect_drift(manifest, relay_repo))


def test_new_outcome_after_capture_is_reported_as_source_drift(relay_repo):
    manifest = rm.build_manifest(relay_repo)
    _write_json(
        relay_repo / ".agent_relay" / "session_outcomes" / "sess-new.json",
        {"session_id": "sess-new", "feature_id": "alpha", "status": "blocked"},
    )
    assert any("sess-new.json" in item for item in rm.detect_drift(manifest, relay_repo))


def test_outcome_absolute_paths_are_sanitized_but_source_checksum_is_retained(relay_repo):
    outcome_path = relay_repo / ".agent_relay" / "session_outcomes" / "sess-1.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["prompt_path"] = str(relay_repo / ".agent_relay" / "prompts" / "sess-1.md")
    outcome["debug"] = {"log_path": "/Users/alice/private/session.log"}
    _write_json(outcome_path, outcome)
    _git(relay_repo, "add", ".")
    _git(relay_repo, "commit", "-q", "-m", "chore: include legacy prompt path")

    manifest = rm.build_manifest(relay_repo)
    captured = manifest.session_outcomes[0]
    assert captured["prompt_path"] == ".agent_relay/prompts/sess-1.md"
    assert captured["source_record"]["debug"]["log_path"].startswith("<redacted-absolute-path:")
    assert "/Users/alice/" not in manifest.dump()
    assert outcome_path.relative_to(relay_repo).as_posix() in manifest.checksums
    assert any("absolute host paths were redacted" in warning for warning in manifest.warnings)


def test_drifted_manifest_makes_no_target_change(relay_repo, target_project):
    manifest = rm.build_manifest(relay_repo)
    (relay_repo / "agent-progress.txt").write_text("## Session 9\nrewritten\n", encoding="utf-8")
    before = _tree_snapshot(target_project)
    with pytest.raises(rm.StaleManifestError):
        rm.assert_fresh(manifest, relay_repo)
    assert _tree_snapshot(target_project) == before


def test_interrupted_import_resumes_by_rerun(relay_repo, target_project, monkeypatch):
    manifest = rm.build_manifest(relay_repo)
    real_write = rm._atomic_write
    calls = {"n": 0}

    def flaky(path: Path, text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated interruption")
        real_write(path, text)

    monkeypatch.setattr(rm, "_atomic_write", flaky)
    with pytest.raises(OSError):
        rm.apply_import(manifest, target_project, apply=True, source=relay_repo)

    monkeypatch.setattr(rm, "_atomic_write", real_write)
    assert not (target_project / rm.RECEIPT_DIR / manifest.migration_id / rm.RECEIPT_FILE).exists()
    plan = rm.apply_import(manifest, target_project, apply=True, source=relay_repo)
    assert plan["new_tasks"] == [] and len(plan["skipped_tasks"]) == 6
    tasks = rm._backlog_module().load(target_project).tasks
    assert len(tasks) == 6 and len({t.id for t in tasks}) == 6
    assert all(t.state != rm._backlog_module().DONE for t in tasks)


def test_source_repository_is_never_modified(relay_repo, target_project):
    before_head = _git(relay_repo, "rev-parse", "HEAD")
    before_status = _git(relay_repo, "status", "--porcelain")
    before_tree = {p.relative_to(relay_repo): p.read_bytes() for p in relay_repo.rglob("*") if p.is_file()}

    rm.apply_import(rm.build_manifest(relay_repo), target_project, apply=True, source=relay_repo)

    assert _git(relay_repo, "rev-parse", "HEAD") == before_head
    assert _git(relay_repo, "status", "--porcelain") == before_status
    assert {p.relative_to(relay_repo): p.read_bytes() for p in relay_repo.rglob("*") if p.is_file()} == before_tree


def test_same_repository_cannot_be_source_and_target(relay_repo):
    with pytest.raises(rm.MigrationError):
        rm.apply_import(rm.build_manifest(relay_repo), relay_repo, apply=True, source=relay_repo)
