import json
from pathlib import Path

from langchain_code_agent.evals.runner import (
    _first_failure_code,
    _first_failure_origin,
    _first_failure_stage,
    load_eval_case,
    run_eval_case,
    run_eval_suite,
)
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.result import AttemptResult, ErrorContext, FinalReport, RunResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "agent_tasks"


def test_load_eval_case_reads_sample_definition() -> None:
    case = load_eval_case(CASE_DIR / "create_file_success.json")

    assert case.id == "create-file-success"
    assert case.expected_success is True
    assert case.plans[0].steps[0].action == "write_file"


def test_load_eval_case_supports_real_planner_sample_metadata() -> None:
    case = load_eval_case(CASE_DIR / "real_planner" / "create_snake_game.json")

    assert case.id == "real-planner-create-snake-game"
    assert case.planner_backend == "langchain"
    assert case.plans == []
    assert case.expected_files[0].content_contains == ["game.js", "style.css"]


def test_run_eval_case_checks_expected_file_state(tmp_path: Path) -> None:
    case = load_eval_case(CASE_DIR / "create_file_success.json")

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is True
    assert result.schema_version == "eval-case-result-v4"
    assert result.run_id
    assert result.artifact_path is not None
    assert Path(result.artifact_path).exists()
    assert result.actions == ["write_file"]
    assert result.failure_reasons == []


def test_run_eval_case_checks_expected_file_contains(tmp_path: Path) -> None:
    real_case = load_eval_case(CASE_DIR / "real_planner" / "update_readme_section.json")
    case = real_case.model_copy(
        update={
            "planner_backend": "noop",
            "plans": [
                Plan(
                    summary="Update README.",
                    steps=[
                        PlanStep(
                            action="insert_text",
                            description="Add the eval harness section.",
                            arguments={
                                "path": "README.md",
                                "anchor": "# Demo Project\n",
                                "text": (
                                    "\n## Agent Eval Harness\n"
                                    "stable regression safety net\n"
                                ),
                                "position": "after",
                            },
                        )
                    ],
                )
            ],
        }
    )

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is True


def test_failure_mode_detects_failed_tests(tmp_path: Path) -> None:
    case = load_eval_case(CASE_DIR / "failure_modes" / "detects_failed_tests.json")

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is False
    assert result.observed_failure_type == "tool_error"
    assert result.error_types == ["ToolExecutionError"]


def test_failure_mode_detects_unexpected_file_change(tmp_path: Path) -> None:
    case = load_eval_case(
        CASE_DIR / "failure_modes" / "detects_unexpected_file_change.json"
    )

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is False
    assert result.observed_failure_type == "completion_failure"
    assert result.error_types == ["IncompleteTaskResult"]


def test_failure_mode_classifies_provider_timeout(tmp_path: Path) -> None:
    case = load_eval_case(
        CASE_DIR / "failure_modes" / "classifies_provider_timeout.json"
    )

    result = run_eval_case(
        case,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path,
    )

    assert result.passed is True
    assert result.agent_success is False
    assert result.failure_code == "provider_timeout"
    assert result.failure_origin == "model_service"
    assert result.observed_failure_type == "planning_failure"


def test_run_eval_suite_generates_baseline_report(tmp_path: Path) -> None:
    case_paths = sorted(CASE_DIR.glob("*.json"))
    report_path = tmp_path / "baseline.json"

    report = run_eval_suite(
        case_paths,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path / "workspaces",
        report_path=report_path,
    )

    assert report.schema_version == "eval-report-v4"
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
    assert report.model_service_failure_rate == 0.0
    assert report.agent_capability_failure_rate == 1 / 7
    assert report.failure_origins == {"agent_capability": 1}
    assert report.planning_failure_codes == {"missing_workspace_path": 1}
    assert report.planning_failure_origins == {"agent_capability": 1}
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
    replanned_success = next(
        result
        for result in report.case_results
        if result.id == "replan-after-completion-failure"
    )
    assert replanned_success.agent_success is True
    assert replanned_success.failure_code is None
    assert replanned_success.failure_origin is None


def test_run_eval_suite_tracks_model_service_failures_separately(tmp_path: Path) -> None:
    case_paths = [
        CASE_DIR / "missing_file_rejected.json",
        CASE_DIR / "failure_modes" / "classifies_provider_timeout.json",
    ]

    report = run_eval_suite(
        case_paths,
        project_root=PROJECT_ROOT,
        workspaces_root=tmp_path / "workspaces",
    )

    assert report.total_cases == 2
    assert report.passed_cases == 2
    assert report.failure_codes == {
        "missing_workspace_path": 1,
        "provider_timeout": 1,
    }
    assert report.failure_origins == {
        "agent_capability": 1,
        "model_service": 1,
    }
    assert report.planning_failure_origins == {
        "agent_capability": 1,
        "model_service": 1,
    }
    assert report.agent_capability_failure_rate == 0.5
    assert report.model_service_failure_rate == 0.5


def test_failure_helpers_use_terminal_failure_context() -> None:
    result = RunResult(
        task="demo",
        workspace_root=str(PROJECT_ROOT),
        execution_mode="execute",
        planner="langchain",
        plan=Plan(summary="demo", steps=[]),
        events=[],
        step_results=[],
        final_report=FinalReport(
            success=False,
            task_input={"task": "demo"},
            plan_summary="demo",
            total_steps=0,
            successful_steps=0,
            failed_steps=0,
            planned_steps=0,
            errors=[
                ErrorContext(
                    error_type="PlanValidationError",
                    message="first",
                    stage="planner_call",
                    failure_code="invalid_action_arguments",
                    failure_origin="agent_capability",
                ),
                ErrorContext(
                    error_type="ResponseValidationError",
                    message="second",
                    stage="planner_call",
                    failure_code="provider_timeout",
                    failure_origin="model_service",
                ),
            ],
        ),
        attempts=[
            AttemptResult(
                attempt=1,
                task="demo",
                plan=Plan(summary="demo", steps=[]),
                step_results=[],
                success=False,
            )
        ],
    )

    assert _first_failure_stage(result) == "planner_call"
    assert _first_failure_code(result) == "provider_timeout"
    assert _first_failure_origin(result) == "model_service"
