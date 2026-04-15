from __future__ import annotations

from pathlib import Path

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.tools.run_shell import run_argv_tool


def run_command_tool(
    *,
    argv: list[str],
    workspace_root: Path,
    timeout_seconds: int,
    allowed_commands: list[str],
    working_directory: str | None = None,
) -> ToolResult:
    return run_argv_tool(
        argv=argv,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
        allowed_commands=allowed_commands,
        working_directory=working_directory,
        command=" ".join(argv),
    )
