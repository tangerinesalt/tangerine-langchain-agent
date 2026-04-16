from pathlib import Path

from langchain_code_agent.actions import (
    ActionRuntime,
    action_argument_schemas_text,
    action_langchain_specs,
    action_names,
    action_produces_shell_output,
    execute_action,
    validate_action_arguments,
)
from langchain_code_agent.workspace.repository import Repository


def test_action_names_include_expected_entries() -> None:
    names = action_names()

    assert "get_current_date" in names
    assert "write_file" in names
    assert "run_tests" in names


def test_validate_action_arguments_uses_registry_rules() -> None:
    missing = validate_action_arguments("write_file", {"path": "notes.txt"})
    unknown = validate_action_arguments("list_files", {"limit": 10, "path": "."})

    assert "missing required arguments" in str(missing)
    assert "does not accept arguments" in str(unknown)


def test_action_argument_schemas_text_is_generated_from_registry() -> None:
    schemas = action_argument_schemas_text()

    assert '- get_current_date: {}' in schemas
    assert (
        '- move_file: {"source_path": string required, "destination_path": string required}'
        in schemas
    )
    assert (
        '- run_python_script: {"script": string required, "working_directory": string optional}'
        in schemas
    )


def test_action_langchain_specs_include_tool_metadata() -> None:
    specs = {spec.name: spec for spec in action_langchain_specs()}

    assert "write_file" in specs
    assert specs["write_file"].langchain_args_schema is not None
    assert "UTF-8 text file" in str(specs["write_file"].langchain_description)


def test_execute_action_runs_registry_executor(tmp_path: Path) -> None:
    repository = Repository(tmp_path, ignore_patterns=[])
    runtime = ActionRuntime(
        repository=repository,
        workspace_root=tmp_path,
        shell_timeout_seconds=5,
        allowed_shell_commands=["python"],
    )

    result = execute_action(
        "write_file",
        runtime,
        {"path": "notes.txt", "content": "hello", "overwrite": False},
    )

    assert result.ok is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"


def test_action_produces_shell_output_marks_command_actions() -> None:
    assert action_produces_shell_output("run_command") is True
    assert action_produces_shell_output("run_tests") is True
    assert action_produces_shell_output("write_file") is False
