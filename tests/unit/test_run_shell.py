from pathlib import Path

from langchain_code_agent.tools.run_shell import run_shell_tool


def test_run_shell_tool_executes_command(tmp_path: Path) -> None:
    result = run_shell_tool(
        command='python -c "print(\'hello\')"',
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is True
    assert "hello" in result.data["stdout"]


def test_run_shell_tool_blocks_non_whitelisted_command(tmp_path: Path) -> None:
    result = run_shell_tool(
        command="git status",
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is False
    assert "not allowed" in str(result.error)


def test_run_shell_tool_blocks_working_directory_escape(tmp_path: Path) -> None:
    result = run_shell_tool(
        command='python -c "print(\'hello\')"',
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
        working_directory="../",
    )

    assert result.ok is False
    assert "escapes workspace root" in str(result.error)


def test_run_shell_tool_enforces_timeout(tmp_path: Path) -> None:
    result = run_shell_tool(
        command='python -c "import time; time.sleep(2)"',
        workspace_root=tmp_path,
        timeout_seconds=1,
        allowed_commands=["python"],
    )

    assert result.ok is False
    assert "timed out" in str(result.error)
