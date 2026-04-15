from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Task:
    goal: str
    workspace_root: Path
    execution_mode: str
