from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.tools.delete_file import delete_file_tool
from langchain_code_agent.tools.find_files_by_name import find_files_by_name_tool
from langchain_code_agent.tools.get_current_date import get_current_date_tool
from langchain_code_agent.tools.glob_files import glob_files_tool
from langchain_code_agent.tools.insert_text import insert_text_tool
from langchain_code_agent.tools.list_files import list_files_tool
from langchain_code_agent.tools.move_file import move_file_tool
from langchain_code_agent.tools.read_file import read_file_tool
from langchain_code_agent.tools.read_file_head import read_file_head_tool
from langchain_code_agent.tools.replace_in_file import replace_in_file_tool
from langchain_code_agent.tools.run_command import run_command_tool
from langchain_code_agent.tools.run_python_script import run_python_script_tool
from langchain_code_agent.tools.run_shell import run_shell_tool
from langchain_code_agent.tools.run_tests import run_tests_tool
from langchain_code_agent.tools.search_text import search_text_tool
from langchain_code_agent.tools.tree_view import tree_view_tool
from langchain_code_agent.tools.write_file import write_file_tool
from langchain_code_agent.workspace.repository import Repository


ActionExecutor = Callable[["ActionRuntime", dict[str, object]], ToolResult]


@dataclass(slots=True)
class ActionRuntime:
    repository: Repository
    workspace_root: Path
    shell_timeout_seconds: int
    allowed_shell_commands: list[str]
    test_command: str | None = None


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    executor: ActionExecutor
    allowed_arguments: frozenset[str] = field(default_factory=frozenset)
    required_arguments: frozenset[str] = field(default_factory=frozenset)
    produces_shell_output: bool = False
    planner_arguments_schema: str = "{}"


def action_names() -> list[str]:
    return list(ACTION_SPECS)


def action_names_csv() -> str:
    return ", ".join(action_names())


def action_argument_schemas_text() -> str:
    return "\n".join(
        f'- {spec.name}: {spec.planner_arguments_schema}' for spec in ACTION_SPECS.values()
    )


def get_action_spec(name: str) -> ActionSpec | None:
    return ACTION_SPECS.get(name)


def execute_action(
    name: str,
    runtime: ActionRuntime,
    arguments: dict[str, object],
) -> ToolResult:
    spec = get_action_spec(name)
    if spec is None:
        raise ValueError(f"Unsupported action: {name}")
    return spec.executor(runtime, arguments)


def validate_action_arguments(action: str, arguments: dict[str, object]) -> str | None:
    spec = get_action_spec(action)
    if spec is None:
        return None

    unknown_arguments = sorted(set(arguments) - spec.allowed_arguments)
    if unknown_arguments:
        return (
            f"Action '{action}' does not accept arguments: {', '.join(unknown_arguments)}"
        )

    missing_arguments = sorted(key for key in spec.required_arguments if key not in arguments)
    if missing_arguments:
        return (
            f"Action '{action}' is missing required arguments: {', '.join(missing_arguments)}"
        )
    return None


def action_produces_shell_output(action: str) -> bool:
    spec = get_action_spec(action)
    return False if spec is None else spec.produces_shell_output


def _run_get_current_date(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return get_current_date_tool()


def _run_list_files(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return list_files_tool(runtime.repository, limit=_coerce_int(arguments.get("limit"), 200))


def _run_glob_files(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return glob_files_tool(
        runtime.repository,
        pattern=str(arguments["pattern"]),
        limit=_coerce_int(arguments.get("limit"), 200),
    )


def _run_find_files_by_name(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return find_files_by_name_tool(
        runtime.repository,
        name=str(arguments["name"]),
        limit=_coerce_int(arguments.get("limit"), 200),
    )


def _run_tree_view(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return tree_view_tool(
        runtime.repository,
        path=str(arguments.get("path", ".")),
        depth=_coerce_int(arguments.get("depth"), 2),
    )


def _run_read_file(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return read_file_tool(runtime.repository, path=str(arguments["path"]))


def _run_read_file_head(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return read_file_head_tool(
        runtime.repository,
        path=str(arguments["path"]),
        start_line=_coerce_int(arguments.get("start_line"), 1),
        max_lines=_coerce_int(arguments.get("max_lines"), 200),
    )


def _run_search_text(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return search_text_tool(
        runtime.repository,
        query=str(arguments["query"]),
        max_results=_coerce_int(arguments.get("max_results"), 20),
        case_sensitive=bool(arguments.get("case_sensitive", False)),
        use_regex=bool(arguments.get("use_regex", False)),
        path_glob=_coerce_optional_str(arguments.get("path_glob")),
    )


def _run_replace_in_file(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return replace_in_file_tool(
        runtime.repository,
        path=str(arguments["path"]),
        old_text=str(arguments["old_text"]),
        new_text=str(arguments["new_text"]),
        count=_coerce_int(arguments.get("count"), 1),
    )


def _run_insert_text(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return insert_text_tool(
        runtime.repository,
        path=str(arguments["path"]),
        anchor=str(arguments["anchor"]),
        text=str(arguments["text"]),
        position=str(arguments.get("position", "after")),
    )


def _run_delete_file(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return delete_file_tool(runtime.repository, path=str(arguments["path"]))


def _run_move_file(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return move_file_tool(
        runtime.repository,
        source_path=str(arguments["source_path"]),
        destination_path=str(arguments["destination_path"]),
    )


def _run_run_command(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    argv = arguments.get("argv")
    if not isinstance(argv, list):
        raise TypeError("run_command expects argv to be a list of strings")
    return run_command_tool(
        argv=[str(item) for item in argv],
        workspace_root=runtime.workspace_root,
        timeout_seconds=runtime.shell_timeout_seconds,
        allowed_commands=runtime.allowed_shell_commands,
        working_directory=_coerce_optional_str(arguments.get("working_directory")),
    )


def _run_run_python_script(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return run_python_script_tool(
        script=str(arguments["script"]),
        workspace_root=runtime.workspace_root,
        timeout_seconds=runtime.shell_timeout_seconds,
        allowed_commands=runtime.allowed_shell_commands,
        working_directory=_coerce_optional_str(arguments.get("working_directory")),
    )


def _run_run_shell(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return run_shell_tool(
        command=str(arguments["command"]),
        workspace_root=runtime.workspace_root,
        timeout_seconds=runtime.shell_timeout_seconds,
        allowed_commands=runtime.allowed_shell_commands,
        working_directory=_coerce_optional_str(arguments.get("working_directory")),
    )


def _run_run_tests(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return run_tests_tool(
        test_command=runtime.test_command,
        workspace_root=runtime.workspace_root,
        timeout_seconds=runtime.shell_timeout_seconds,
        allowed_commands=runtime.allowed_shell_commands,
        working_directory=_coerce_optional_str(arguments.get("working_directory")),
    )


def _run_write_file(runtime: ActionRuntime, arguments: dict[str, object]) -> ToolResult:
    return write_file_tool(
        runtime.repository,
        path=str(arguments["path"]),
        content=str(arguments["content"]),
        overwrite=bool(arguments.get("overwrite", False)),
    )


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got: {type(value)!r}")


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


ACTION_SPECS: dict[str, ActionSpec] = {
    "get_current_date": ActionSpec(
        name="get_current_date",
        executor=_run_get_current_date,
        planner_arguments_schema="{}",
    ),
    "list_files": ActionSpec(
        name="list_files",
        executor=_run_list_files,
        allowed_arguments=frozenset({"limit"}),
        planner_arguments_schema='{"limit": integer optional}',
    ),
    "glob_files": ActionSpec(
        name="glob_files",
        executor=_run_glob_files,
        allowed_arguments=frozenset({"pattern", "limit"}),
        required_arguments=frozenset({"pattern"}),
        planner_arguments_schema='{"pattern": string required, "limit": integer optional}',
    ),
    "find_files_by_name": ActionSpec(
        name="find_files_by_name",
        executor=_run_find_files_by_name,
        allowed_arguments=frozenset({"name", "limit"}),
        required_arguments=frozenset({"name"}),
        planner_arguments_schema='{"name": string required, "limit": integer optional}',
    ),
    "tree_view": ActionSpec(
        name="tree_view",
        executor=_run_tree_view,
        allowed_arguments=frozenset({"path", "depth"}),
        planner_arguments_schema='{"path": string optional, "depth": integer optional}',
    ),
    "read_file": ActionSpec(
        name="read_file",
        executor=_run_read_file,
        allowed_arguments=frozenset({"path"}),
        required_arguments=frozenset({"path"}),
        planner_arguments_schema='{"path": string required}',
    ),
    "read_file_head": ActionSpec(
        name="read_file_head",
        executor=_run_read_file_head,
        allowed_arguments=frozenset({"path", "start_line", "max_lines"}),
        required_arguments=frozenset({"path"}),
        planner_arguments_schema=(
            '{"path": string required, "start_line": integer optional, '
            '"max_lines": integer optional}'
        ),
    ),
    "search_text": ActionSpec(
        name="search_text",
        executor=_run_search_text,
        allowed_arguments=frozenset(
            {"query", "max_results", "case_sensitive", "use_regex", "path_glob"}
        ),
        required_arguments=frozenset({"query"}),
        planner_arguments_schema=(
            '{"query": string required, "max_results": integer optional, '
            '"case_sensitive": boolean optional, "use_regex": boolean optional, '
            '"path_glob": string optional}'
        ),
    ),
    "replace_in_file": ActionSpec(
        name="replace_in_file",
        executor=_run_replace_in_file,
        allowed_arguments=frozenset({"path", "old_text", "new_text", "count"}),
        required_arguments=frozenset({"path", "old_text", "new_text"}),
        planner_arguments_schema=(
            '{"path": string required, "old_text": string required, '
            '"new_text": string required, "count": integer optional}'
        ),
    ),
    "insert_text": ActionSpec(
        name="insert_text",
        executor=_run_insert_text,
        allowed_arguments=frozenset({"path", "anchor", "text", "position"}),
        required_arguments=frozenset({"path", "anchor", "text"}),
        planner_arguments_schema=(
            '{"path": string required, "anchor": string required, '
            '"text": string required, "position": string optional}'
        ),
    ),
    "delete_file": ActionSpec(
        name="delete_file",
        executor=_run_delete_file,
        allowed_arguments=frozenset({"path"}),
        required_arguments=frozenset({"path"}),
        planner_arguments_schema='{"path": string required}',
    ),
    "move_file": ActionSpec(
        name="move_file",
        executor=_run_move_file,
        allowed_arguments=frozenset({"source_path", "destination_path"}),
        required_arguments=frozenset({"source_path", "destination_path"}),
        planner_arguments_schema=(
            '{"source_path": string required, "destination_path": string required}'
        ),
    ),
    "run_command": ActionSpec(
        name="run_command",
        executor=_run_run_command,
        allowed_arguments=frozenset({"argv", "working_directory"}),
        required_arguments=frozenset({"argv"}),
        produces_shell_output=True,
        planner_arguments_schema=(
            '{"argv": string array required, "working_directory": string optional}'
        ),
    ),
    "run_python_script": ActionSpec(
        name="run_python_script",
        executor=_run_run_python_script,
        allowed_arguments=frozenset({"script", "working_directory"}),
        required_arguments=frozenset({"script"}),
        produces_shell_output=True,
        planner_arguments_schema=(
            '{"script": string required, "working_directory": string optional}'
        ),
    ),
    "run_shell": ActionSpec(
        name="run_shell",
        executor=_run_run_shell,
        allowed_arguments=frozenset({"command", "working_directory"}),
        required_arguments=frozenset({"command"}),
        produces_shell_output=True,
        planner_arguments_schema=(
            '{"command": string required, "working_directory": string optional}'
        ),
    ),
    "run_tests": ActionSpec(
        name="run_tests",
        executor=_run_run_tests,
        allowed_arguments=frozenset({"working_directory"}),
        produces_shell_output=True,
        planner_arguments_schema='{"working_directory": string optional}',
    ),
    "write_file": ActionSpec(
        name="write_file",
        executor=_run_write_file,
        allowed_arguments=frozenset({"path", "content", "overwrite"}),
        required_arguments=frozenset({"path", "content"}),
        planner_arguments_schema=(
            '{"path": string required, "content": string required, '
            '"overwrite": boolean optional}'
        ),
    ),
}
