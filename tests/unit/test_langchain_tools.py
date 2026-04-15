from pathlib import Path
from types import SimpleNamespace

from langchain_code_agent.config import AgentConfig
from langchain_code_agent.tools.langchain_tools import (
    LANGCHAIN_TOOL_SPECS,
    build_langchain_tools,
    build_tool_context,
)


def test_langchain_tools_use_runtime_context_and_hide_runtime_arg(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    config = AgentConfig(
        workspace_root=tmp_path,
        shell_timeout_seconds=5,
        allowed_shell_commands=["python"],
        ignore_patterns=[],
    )
    runtime = SimpleNamespace(context=build_tool_context(config))
    tools = {tool.name: tool for tool in build_langchain_tools()}

    date_result = tools["get_current_date"].func(runtime=runtime)
    list_result = tools["list_files"].func(limit=10, runtime=runtime)
    glob_result = tools["glob_files"].func(pattern="*.py", limit=10, runtime=runtime)
    find_result = tools["find_files_by_name"].func(name="main", limit=10, runtime=runtime)
    tree_result = tools["tree_view"].func(path=".", depth=2, runtime=runtime)
    head_result = tools["read_file_head"].func(
        path="main.py",
        start_line=1,
        max_lines=1,
        runtime=runtime,
    )
    search_result = tools["search_text"].func(
        query="hello",
        max_results=5,
        case_sensitive=False,
        use_regex=False,
        path_glob="*.py",
        runtime=runtime,
    )
    shell_result = tools["run_shell"].func(
        command='python -c "print(\'ok\')"',
        working_directory=None,
        runtime=runtime,
    )
    write_result = tools["write_file"].func(
        path="notes.txt",
        content="hello",
        overwrite=False,
        runtime=runtime,
    )
    replace_result = tools["replace_in_file"].func(
        path="notes.txt",
        old_text="hello",
        new_text="world",
        count=1,
        runtime=runtime,
    )
    insert_result = tools["insert_text"].func(
        path="notes.txt",
        anchor="world",
        text="!\n",
        position="after",
        runtime=runtime,
    )
    move_result = tools["move_file"].func(
        source_path="notes.txt",
        destination_path="nested/notes.txt",
        runtime=runtime,
    )
    command_result = tools["run_command"].func(
        argv=["python", "-c", "print('argv')"],
        working_directory=None,
        runtime=runtime,
    )
    python_script_result = tools["run_python_script"].func(
        script=(
            "from pathlib import Path\n"
            "Path('script_note.txt').write_text('ok', encoding='utf-8')\n"
            "print('script')\n"
        ),
        working_directory=None,
        runtime=runtime,
    )

    assert "runtime" not in tools["run_shell"].args
    assert date_result["ok"] is True
    assert "current_date" in date_result["data"]
    assert list_result["ok"] is True
    assert glob_result["data"]["count"] == 1
    assert find_result["data"]["count"] == 1
    assert tree_result["ok"] is True
    assert head_result["data"]["content"] == "print('hello')"
    assert list_result["data"]["count"] == 1
    assert search_result["data"]["count"] == 1
    assert shell_result["ok"] is True
    assert write_result["ok"] is True
    assert replace_result["ok"] is True
    assert insert_result["ok"] is True
    assert move_result["ok"] is True
    assert command_result["ok"] is True
    assert python_script_result["ok"] is True
    assert (tmp_path / "nested" / "notes.txt").read_text(encoding="utf-8") == "world!\n"
    assert (tmp_path / "script_note.txt").read_text(encoding="utf-8") == "ok"
    assert "ok" in shell_result["data"]["stdout"]
    assert "argv" in command_result["data"]["stdout"]
    assert "script" in python_script_result["data"]["stdout"]


def test_langchain_tools_can_run_tests_with_runtime_context(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    config = AgentConfig(
        workspace_root=tmp_path,
        shell_timeout_seconds=10,
        allowed_shell_commands=["python", "pytest"],
        ignore_patterns=[],
        test_command="python -m pytest -q",
    )
    runtime = SimpleNamespace(context=build_tool_context(config))
    tools = {tool.name: tool for tool in build_langchain_tools()}

    result = tools["run_tests"].func(working_directory=None, runtime=runtime)

    assert result["ok"] is True
    assert "1 passed" in result["data"]["stdout"]


def test_langchain_tools_are_built_from_declared_specs() -> None:
    tools = build_langchain_tools()

    assert [tool.name for tool in tools] == [spec.name for spec in LANGCHAIN_TOOL_SPECS]
