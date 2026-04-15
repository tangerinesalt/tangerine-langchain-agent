from __future__ import annotations

from langchain_code_agent.models.result import ErrorContext, RunResult


MATERIAL_OUTPUT_KEYWORDS = (
    "add",
    "build",
    "create",
    "delete",
    "edit",
    "fix",
    "generate",
    "implement",
    "modify",
    "move",
    "remove",
    "rename",
    "update",
    "write",
)

READ_ONLY_KEYWORDS = (
    "analyze",
    "check",
    "find",
    "inspect",
    "list",
    "read",
    "review",
    "search",
    "show",
    "view",
)


def validate_completion(run_result: RunResult) -> list[ErrorContext]:
    if run_result.execution_mode != "execute":
        return []
    if any(not step.ok for step in run_result.step_results):
        return []
    if not _goal_requires_material_output(run_result.task):
        return []
    if any(step.file_changes for step in run_result.step_results):
        return []
    return [
        ErrorContext(
            error_type="IncompleteTaskResult",
            message=(
                "Task goal suggests creating or changing outputs, but the run produced "
                "no file changes."
            ),
        )
    ]


def _goal_requires_material_output(task_text: str) -> bool:
    lowered = task_text.lower()
    if any(keyword in lowered for keyword in MATERIAL_OUTPUT_KEYWORDS):
        return True
    if any(keyword in lowered for keyword in READ_ONLY_KEYWORDS):
        return False
    return False
