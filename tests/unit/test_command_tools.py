from pathlib import Path

from langchain_code_agent.tools.run_command import run_command_tool
from langchain_code_agent.tools.run_python_script import run_python_script_tool


def test_run_command_tool_executes_argv_command(tmp_path: Path) -> None:
    result = run_command_tool(
        argv=["python", "-c", "print('argv')"],
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is True
    assert result.data["argv"] == ["python", "-c", "print('argv')"]
    assert "argv" in result.data["stdout"]


def test_run_command_tool_blocks_working_directory_escape(tmp_path: Path) -> None:
    result = run_command_tool(
        argv=["python", "-c", "print('argv')"],
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
        working_directory="../",
    )

    assert result.ok is False
    assert "escapes workspace root" in str(result.error)


def test_run_python_script_tool_executes_multiline_script(tmp_path: Path) -> None:
    result = run_python_script_tool(
        script=(
            "from pathlib import Path\n"
            "Path('created.txt').write_text('ok', encoding='utf-8')\n"
            "print('script-ran')\n"
        ),
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["python"],
    )

    assert result.ok is True
    assert "script-ran" in result.data["stdout"]
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "ok"
    assert "script_path" in result.data
    assert not Path(result.data["script_path"]).exists()


def test_run_python_script_tool_blocks_non_whitelisted_python(tmp_path: Path) -> None:
    result = run_python_script_tool(
        script="print('blocked')\n",
        workspace_root=tmp_path,
        timeout_seconds=5,
        allowed_commands=["pytest"],
    )

    assert result.ok is False
    assert "not allowed" in str(result.error)
