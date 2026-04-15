from __future__ import annotations

import re
import subprocess
from pathlib import Path

from langchain_code_agent.tools.base import ToolResult


def run_shell_tool(
    *,
    command: str,
    workspace_root: Path,
    timeout_seconds: int,
    allowed_commands: list[str],
    working_directory: str | None = None,
) -> ToolResult:
    try:
        argv = _parse_command(command)
    except ValueError as exc:
        return ToolResult(ok=False, error=str(exc))

    return run_argv_tool(
        argv=argv,
        workspace_root=workspace_root,
        timeout_seconds=timeout_seconds,
        allowed_commands=allowed_commands,
        working_directory=working_directory,
        command=command,
    )


def run_argv_tool(
    *,
    argv: list[str],
    workspace_root: Path,
    timeout_seconds: int,
    allowed_commands: list[str],
    working_directory: str | None = None,
    command: str | None = None,
) -> ToolResult:
    try:
        executable = _validate_allowed_command(argv, allowed_commands)
        cwd = _resolve_working_directory(workspace_root, working_directory)
    except ValueError as exc:
        return ToolResult(ok=False, error=str(exc))

    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(ok=False, error=f"Command timed out after {timeout_seconds}s: {exc.cmd}")

    return ToolResult(
        ok=completed.returncode == 0,
        data={
            "command": command or " ".join(argv),
            "argv": argv,
            "working_directory": str(cwd),
            "executable": executable,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        error=(
            None
            if completed.returncode == 0
            else f"Command failed with exit code {completed.returncode}"
        ),
    )


def _parse_command(command: str) -> list[str]:
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command.strip())
    argv = [_strip_quotes(token) for token in tokens]
    if not argv:
        raise ValueError("Command cannot be empty.")
    return argv


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def _validate_allowed_command(argv: list[str], allowed_commands: list[str]) -> str:
    executable = Path(argv[0]).name.lower()
    executable = executable.removesuffix(".exe").removesuffix(".cmd").removesuffix(".bat")
    allowed = {item.lower() for item in allowed_commands}
    if executable not in allowed:
        raise ValueError(
            f"Command '{argv[0]}' is not allowed. Allowed commands: {', '.join(sorted(allowed))}"
        )
    return executable


def _resolve_working_directory(workspace_root: Path, working_directory: str | None) -> Path:
    root = workspace_root.resolve()
    target = root if working_directory is None else (root / working_directory).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Working directory escapes workspace root: {working_directory}")
    if not target.exists():
        raise ValueError(f"Working directory does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"Working directory is not a directory: {target}")
    return target
