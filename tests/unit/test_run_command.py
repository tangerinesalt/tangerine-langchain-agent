from pathlib import Path

from langchain_code_agent.tools.run_command import run_command_tool
from langchain_code_agent.tools.run_python_script import run_python_script_tool


def test_run_command_tool_executes_argv(tmp_path: Path) -> None:
    result = run_command_tool(
        argv=["python", "-c", "print('hello')"],
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is True
    assert "hello" in result.data["stdout"]


def test_run_python_script_tool_executes_multiline_script(tmp_path: Path) -> None:
    result = run_python_script_tool(
        script=(
            "from pathlib import Path\n"
            "Path('note.txt').write_text('hello', encoding='utf-8')\n"
            "print('done')\n"
        ),
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is True
    assert "done" in result.data["stdout"]
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
