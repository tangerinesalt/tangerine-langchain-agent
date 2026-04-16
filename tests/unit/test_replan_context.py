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
    assert context.successful_actions == ["list_files"]
    assert context.file_changes == ["notes.txt"]
    assert context.completion_failures == ["Expected file to exist after run: notes.txt"]
    assert context.failed_steps[0].action == "read_file_head"
