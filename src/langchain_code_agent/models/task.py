from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_code_agent.models.replan import ReplanContext


@dataclass(slots=True)
class Task:
    goal: str
    workspace_root: Path
    execution_mode: str
    replan_context: ReplanContext | None = None
