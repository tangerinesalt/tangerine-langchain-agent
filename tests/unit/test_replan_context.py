from langchain_code_agent.agent.replan_context import build_replan_context
from langchain_code_agent.models.plan import CompletionCheck, Plan, PlanStep
from langchain_code_agent.models.result import (
    AttemptResult,
    ErrorContext,
    FileChange,
    StepExecutionResult,
)


def test_build_replan_context_collects_failures_successes_and_file_changes() -> None:
    attempt_result = AttemptResult(
        attempt=1,
        task="write notes.txt",
        plan=Plan(
            summary="Inspect first.",
            steps=[
                PlanStep(
                    action="list_files",
                    description="Inspect files.",
                    arguments={"limit": 20},
                ),
            ],
            completion_checks=[
                CompletionCheck(check_type="file_exists", arguments={"path": "notes.txt"})
            ],
        ),
        step_results=[
            StepExecutionResult(
                action="list_files",
                status="completed",
                ok=True,
                arguments={"limit": 20},
                file_changes=[FileChange(path="notes.txt", change_type="added")],
            ),
            StepExecutionResult(
                action="read_file_head",
                status="failed",
                ok=False,
                arguments={"path": "pytest.log"},
                error="Path does not exist: pytest.log",
                error_context=ErrorContext(
                    error_type="ToolExecutionError",
                    message="Path does not exist: pytest.log",
                    action="read_file_head",
                    arguments={"path": "pytest.log"},
                    step_index=2,
                ),
            ),
        ],
        success=False,
        errors=[
            ErrorContext(
                error_type="IncompleteTaskResult",
                message="Expected file to exist after run: notes.txt",
                arguments={"path": "notes.txt"},
            )
        ],
        completion_errors=[
            ErrorContext(
                error_type="IncompleteTaskResult",
                message="Expected file to exist after run: notes.txt",
                arguments={"path": "notes.txt"},
            )
        ],
    )

    context = build_replan_context("write notes.txt", attempt_result)

    assert context.original_task == "write notes.txt"
    assert context.attempt == 1
    assert context.previous_plan_summary == "Inspect first."
    assert context.attempt_failures == []
    assert context.successful_actions == ["list_files"]
    assert context.file_changes == ["notes.txt"]
    assert context.completion_failures == ["Expected file to exist after run: notes.txt"]
    assert context.failed_steps[0].action == "read_file_head"


def test_build_replan_context_includes_failed_shell_output_excerpts() -> None:
    attempt_result = AttemptResult(
        attempt=2,
        task="fix failing tests",
        plan=Plan(
            summary="Run tests.",
            steps=[
                PlanStep(
                    action="run_tests",
                    description="Run tests.",
                    arguments={},
                )
            ],
        ),
        step_results=[
            StepExecutionResult(
                action="run_tests",
                status="failed",
                ok=False,
                arguments={},
                data={
                    "stdout": (
                        "FAILED tests/test_math_utils.py::test_scale_and_offset\n"
                        "assert 7 == 12"
                    ),
                    "stderr": "traceback line",
                },
                error="Command failed with exit code 1",
                error_context=ErrorContext(
                    error_type="ToolExecutionError",
                    message="Command failed with exit code 1",
                    action="run_tests",
                    arguments={},
                    step_index=1,
                ),
            ),
        ],
        success=False,
    )

    context = build_replan_context("fix failing tests", attempt_result)

    assert context.failed_steps[0].action == "run_tests"
    assert "test_scale_and_offset" in str(context.failed_steps[0].stdout_excerpt)
    assert context.failed_steps[0].stderr_excerpt == "traceback line"


def test_build_replan_context_includes_non_completion_attempt_failures() -> None:
    attempt_result = AttemptResult(
        attempt=1,
        task="fix failing tests",
        plan=Plan(summary="Planning failed.", steps=[]),
        step_results=[],
        success=False,
        errors=[
            ErrorContext(
                error_type="ValueError",
                message="Fix-failing-tests tasks must include at least one edit step.",
            )
        ],
    )

    context = build_replan_context("fix failing tests", attempt_result)

    assert context.failed_steps == []
    assert context.attempt_failures == [
        "Fix-failing-tests tasks must include at least one edit step."
    ]
