import json
from pathlib import Path

from langchain_code_agent.evals.runner import load_eval_case, run_eval_case, run_eval_suite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "agent_tasks"


def test_load_eval_case_reads_sample_definition() -> None:
    case = load_eval_case(CASE_DIR / "create_file_success.json")

    assert case.id == "create-file-success"
    assert case.expected_success is True
    assert case.plans[0].steps[0].action == "write_file"


def test_run_eval_case_checks_expected_file_state(tmp_path: Path) -> None:
    case = load_eval_case(CASE_DIR / "create_file_success.json")

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is True
    assert result.schema_version == "eval-case-result-v3"
    assert result.run_id
    assert result.artifact_path is not None
    assert Path(result.artifact_path).exists()
    assert result.actions == ["write_file"]
    assert result.failure_reasons == []


def test_run_eval_suite_generates_baseline_report(tmp_path: Path) -> None:
    case_paths = sorted(CASE_DIR.glob("*.json"))
    report_path = tmp_path / "baseline.json"

    report = run_eval_suite(
        case_paths,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path / "workspaces",
        report_path=report_path,
    )

    assert report.schema_version == "eval-report-v3"
    assert report.total_cases == 7
    assert report.passed_cases == 7
    assert report.failed_cases == 0
    assert report.success_rate == 1.0
    assert report.replan_rate == 1 / 7
    assert report.tool_error_rate == 1 / 7
    assert report.completion_failure_rate == 2 / 7
    assert report.false_success_rate == 1 / 7
    assert report.incomplete_task_rate == 1 / 7
    assert report.planning_failure_rate == 1 / 7
    assert report.plan_repair_success_rate == 1.0
    assert report.planning_failure_codes == {"missing_workspace_path": 1}
    assert report.repair_codes == {"append_run_tests_verification": 1}
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["total_cases"] == 7
    missing_file = next(
        result for result in report.case_results if result.id == "missing-file-rejected"
    )
    assert missing_file.observed_failure_type == "planning_failure"
    assert missing_file.failure_stage == "validate_plan"
    assert missing_file.failure_code == "missing_workspace_path"
    assert missing_file.artifact_path is not None
    repaired = next(
        result for result in report.case_results if result.id == "fix-tests-adds-verification"
    )
    assert repaired.planning_repaired is True
    assert repaired.repair_code == "append_run_tests_verification"
    incomplete = next(
        result
        for result in report.case_results
        if result.id == "file-contains-detects-incomplete"
    )
    assert incomplete.agent_success is False
    assert incomplete.observed_failure_type == "completion_failure"
