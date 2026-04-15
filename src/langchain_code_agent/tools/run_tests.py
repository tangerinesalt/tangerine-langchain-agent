from __future__ import annotations

from pathlib import Path

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.tools.run_shell import run_shell_tool


def run_tests_tool(
    *,
    test_command: str | None,
    workspace_root: Path,
    timeout_seconds: int,
    allowed_commands: list[str],
    working_directory: str | None = None,
) -> ToolResult:
    if not test_command:
        return ToolResult(ok=False, error="No test command configured.")
    return run_shell_tool(
        command=test_command,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
        allowed_commands=allowed_commands,
        working_directory=working_directory,
    )
