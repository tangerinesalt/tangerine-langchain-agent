from __future__ import annotations

from typing import Any

from langchain_code_agent.models.task import Task


def build_task_input(task: Task, planner_backend: str) -> dict[str, Any]:
    return {
        "task": task.goal,
        "workspace_root": str(task.workspace_root),
        "execution_mode": task.execution_mode,
        "planner_backend": planner_backend,
    }

