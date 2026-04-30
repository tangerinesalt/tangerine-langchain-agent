from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_code_agent.models.planning_context import PlanningContext
from langchain_code_agent.models.replan import ReplanContext


@dataclass(slots=True)
class Task:
    goal: str
    workspace_root: Path
    execution_mode: str
    planning_context: PlanningContext | None = None
    replan_context: ReplanContext | None = None
