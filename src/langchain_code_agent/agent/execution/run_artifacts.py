from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_code_agent.models.result import RunResult


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def run_artifact_path(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / ".lca" / "runs" / run_id / "result.json"


def write_run_artifact(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

