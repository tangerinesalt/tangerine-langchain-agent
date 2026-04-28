import json
from pathlib import Path

from langchain_code_agent.evals.experience import (
    archive_eval_suite,
    load_experience_records,
    query_experience_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "agent_tasks"


def test_archive_eval_suite_writes_queryable_experience_records(tmp_path: Path) -> None:
    case_paths = [
        CASE_DIR / "create_file_success.json",
        CASE_DIR / "fix_tests_adds_verification.json",
        CASE_DIR / "missing_file_rejected.json",
    ]

    archive = archive_eval_suite(
        case_paths,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path / "workspaces",
        archive_dir=tmp_path / "experience",
        report_path=tmp_path / "report.json",
    )

    assert archive.report.total_cases == 3
    assert archive.report.passed_cases == 3
    assert archive.paths is not None
    records_path = Path(archive.paths.records_path)
    index_path = Path(archive.paths.index_path)
    assert records_path.exists()
    assert index_path.exists()

    records = load_experience_records(records_path)
    assert len(records) == 3
    assert archive.index.record_count == 3
    assert len(archive.index.by_outcome["success"]) == 2
    assert len(archive.index.by_outcome["expected_failure"]) == 1
    assert "write_file" in archive.index.by_action
    assert "missing_workspace_path" in archive.index.by_failure_code
    assert "append_run_tests_verification" in archive.index.by_repair_code

    missing_file = query_experience_records(
        records,
        failure_code="missing_workspace_path",
    )
    assert [record.case_id for record in missing_file] == ["missing-file-rejected"]
    assert missing_file[0].observed_failure_type == "planning_failure"

    repaired = query_experience_records(
        records,
        repair_code="append_run_tests_verification",
    )
    assert [record.case_id for record in repaired] == ["fix-tests-adds-verification"]
    assert repaired[0].repair is not None
    assert repaired[0].repair.problem_code == "missing_validation_step"
    assert repaired[0].repair.agent_success_after_repair is True
    assert repaired[0].repair.actions_after_repair == ["write_file", "run_tests"]

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == "agent-experience-index-v1"
