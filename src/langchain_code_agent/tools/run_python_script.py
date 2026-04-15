from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.tools.run_shell import run_argv_tool


def run_python_script_tool(
    *,
    script: str,
    workspace_root: Path,
    timeout_seconds: int,
    allowed_commands: list[str],
    working_directory: str | None = None,
) -> ToolResult:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            delete=False,
        ) as handle:
            handle.write(script)
            temp_path = handle.name

        result = run_argv_tool(
            argv=["python", temp_path],
            workspace_root=workspace_root,
            timeout_seconds=timeout_seconds,
            allowed_commands=allowed_commands,
            working_directory=working_directory,
            command="python <temp-script>",
        )
        if result.ok or result.data:
            result.data["script_path"] = temp_path
        return result
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
