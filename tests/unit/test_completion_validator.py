from pathlib import Path

from langchain_code_agent.agent.completion_validator import derive_completion_checks, validate_completion
from langchain_code_agent.models.plan import CompletionCheck, Plan, PlanStep
from langchain_code_agent.models.result import FinalReport, FileChange, RunResult, StepExecutionResult


def test_derive_completion_checks_from_plan_steps() -> None:
    plan = Plan(
        summary="Write and test.",
        steps=[
            PlanStep(
                action="write_file",
                description="Write notes.",
                arguments={"path": "notes.txt", "content": "hello"},
            ),
            PlanStep(action="run_tests", description="Run tests.", arguments={}),
        ],
    )

    checks = derive_completion_checks(plan)

    assert [check.check_type for check in checks] == ["file_exists", "action_succeeded"]
    assert checks[0].arguments == {"path": "notes.txt"}
    assert checks[1].arguments == {"action": "run_tests"}


def test_validate_completion_uses_explicit_completion_checks() -> None:
    run_result = RunResult(
        task="write notes",
        workspace_root=str(Path.cwd()),
        execution_mode="execute",
        planner="noop",
        plan=Plan(
            summary="Write notes.",
            steps=[],
            completion_checks=[
                CompletionCheck(check_type="file_exists", arguments={"path": "notes.txt"})
            ],
        ),
        events=[],
        step_results=[
            StepExecutionResult(
                action="write_file",
                status="completed",
                ok=True,
                arguments={"path": "notes.txt", "content": "hello"},
                data={},
                file_changes=[FileChange(path="notes.txt", change_type="added")],
            )
        ],
        final_report=FinalReport(
            success=True,
            task_input={},
            plan_summary="Write notes.",
            total_steps=1,
            successful_steps=1,
            failed_steps=0,
            planned_steps=0,
        ),
    )

    assert validate_completion(run_result) == []


def test_validate_completion_reports_failed_shell_output_check() -> None:
    run_result = RunResult(
        task="run tests",
        workspace_root=str(Path.cwd()),
        execution_mode="execute",
        planner="noop",
        plan=Plan(
            summary="Run tests.",
            steps=[],
            completion_checks=[
                CompletionCheck(
                    check_type="shell_output_contains",
                    arguments={"action": "run_tests", "text": "2 passed"},
                )
            ],
        ),
        events=[],
        step_results=[
            StepExecutionResult(
                action="run_tests",
                status="completed",
                ok=True,
                arguments={},
                data={"stdout": "1 passed"},
                file_changes=[],
            )
        ],
        final_report=FinalReport(
            success=True,
            task_input={},
            plan_summary="Run tests.",
            total_steps=1,
            successful_steps=1,
            failed_steps=0,
            planned_steps=0,
        ),
    )

    errors = validate_completion(run_result)

    assert len(errors) == 1
    assert errors[0].error_type == "IncompleteTaskResult"
    assert "2 passed" in errors[0].message
