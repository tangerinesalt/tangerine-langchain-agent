from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_code_agent.actions import ActionRuntime, execute_action
from langchain_code_agent.config import AgentConfig
from langchain_code_agent.workspace.repository import Repository


@dataclass(slots=True)
class AgentToolContext:
    workspace_root: str
    ignore_patterns: list[str]
    shell_timeout_seconds: int
    allowed_shell_commands: list[str]
    test_command: str | None = None


class ListFilesInput(BaseModel):
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum files to return.")


class GetCurrentDateInput(BaseModel):
    pass


class GlobFilesInput(BaseModel):
    pattern: str = Field(description="Glob pattern relative to the workspace root.")
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum files to return.")


class FindFilesByNameInput(BaseModel):
    name: str = Field(description="Filename fragment to match.")
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum files to return.")


class TreeViewInput(BaseModel):
    path: str = Field(default=".", description="Relative directory path inside the workspace root.")
    depth: int = Field(default=2, ge=0, le=10, description="Directory depth to include.")


class ReadFileInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")


class ReadFileHeadInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")
    start_line: int = Field(default=1, ge=1, description="1-based starting line number.")
    max_lines: int = Field(default=200, ge=1, le=2000, description="Maximum lines to return.")


class SearchTextInput(BaseModel):
    query: str = Field(min_length=1, description="Text to search for in repository files.")
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum matching lines to return.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether search should be case sensitive.",
    )
    use_regex: bool = Field(
        default=False,
        description="Whether query should be treated as a regex.",
    )
    path_glob: str | None = Field(
        default=None,
        description="Optional glob pattern to limit searched files.",
    )


class RunShellInput(BaseModel):
    command: str = Field(description="Allowed shell command to execute.")
    working_directory: str | None = Field(
        default=None,
        description="Optional relative directory inside the workspace root.",
    )


class RunTestsInput(BaseModel):
    working_directory: str | None = Field(
        default=None,
        description="Optional relative directory inside the workspace root.",
    )


class WriteFileInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")
    content: str = Field(description="UTF-8 text content to write.")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite an existing file.",
    )


class ReplaceInFileInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")
    old_text: str = Field(description="Original text to replace.")
    new_text: str = Field(description="Replacement text.")
    count: int = Field(default=1, ge=1, description="Maximum replacements to perform.")


class InsertTextInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")
    anchor: str = Field(description="Anchor text used to determine insertion point.")
    text: str = Field(description="Text to insert.")
    position: str = Field(default="after", description="Insert before or after the anchor.")


class DeleteFileInput(BaseModel):
    path: str = Field(description="Relative file path inside the workspace root.")


class MoveFileInput(BaseModel):
    source_path: str = Field(description="Existing relative file path inside the workspace root.")
    destination_path: str = Field(description="New relative file path inside the workspace root.")


class RunCommandInput(BaseModel):
    argv: list[str] = Field(description="Command argv list. First item must be whitelisted.")
    working_directory: str | None = Field(
        default=None,
        description="Optional relative directory inside the workspace root.",
    )


class RunPythonScriptInput(BaseModel):
    script: str = Field(description="Python source code to execute.")
    working_directory: str | None = Field(
        default=None,
        description="Optional relative directory inside the workspace root.",
    )


@dataclass(frozen=True, slots=True)
class LangChainToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]


LANGCHAIN_TOOL_SPECS: tuple[LangChainToolSpec, ...] = (
    LangChainToolSpec(
        name="get_current_date",
        args_schema=GetCurrentDateInput,
        description="Return the current local date, datetime, timezone, and weekday.",
    ),
    LangChainToolSpec(
        name="list_files",
        args_schema=ListFilesInput,
        description="List repository files under the workspace root. Use this first to inspect layout.",
    ),
    LangChainToolSpec(
        name="glob_files",
        args_schema=GlobFilesInput,
        description="Find files by glob pattern relative to the workspace root.",
    ),
    LangChainToolSpec(
        name="find_files_by_name",
        args_schema=FindFilesByNameInput,
        description="Find files whose names contain the provided fragment.",
    ),
    LangChainToolSpec(
        name="tree_view",
        args_schema=TreeViewInput,
        description="Return a compact directory tree view for a path inside the workspace root.",
    ),
    LangChainToolSpec(
        name="read_file",
        args_schema=ReadFileInput,
        description="Read a text file from the workspace using a relative path.",
    ),
    LangChainToolSpec(
        name="read_file_head",
        args_schema=ReadFileHeadInput,
        description="Read a range of lines from a text file inside the workspace root.",
    ),
    LangChainToolSpec(
        name="search_text",
        args_schema=SearchTextInput,
        description="Search repository text content and return matching lines with file paths.",
    ),
    LangChainToolSpec(
        name="run_shell",
        args_schema=RunShellInput,
        description=(
            "Run a whitelisted shell command within the workspace root. "
            "Use only when file inspection is insufficient."
        ),
    ),
    LangChainToolSpec(
        name="run_tests",
        args_schema=RunTestsInput,
        description="Run the configured test command inside the workspace root.",
    ),
    LangChainToolSpec(
        name="write_file",
        args_schema=WriteFileInput,
        description="Write a UTF-8 text file inside the workspace root.",
    ),
    LangChainToolSpec(
        name="replace_in_file",
        args_schema=ReplaceInFileInput,
        description="Replace text inside an existing workspace file.",
    ),
    LangChainToolSpec(
        name="insert_text",
        args_schema=InsertTextInput,
        description="Insert text before or after an anchor inside an existing workspace file.",
    ),
    LangChainToolSpec(
        name="delete_file",
        args_schema=DeleteFileInput,
        description="Delete an existing file inside the workspace root.",
    ),
    LangChainToolSpec(
        name="move_file",
        args_schema=MoveFileInput,
        description="Move or rename a file inside the workspace root.",
    ),
    LangChainToolSpec(
        name="run_command",
        args_schema=RunCommandInput,
        description="Run a whitelisted command using argv form for safer execution.",
    ),
    LangChainToolSpec(
        name="run_python_script",
        args_schema=RunPythonScriptInput,
        description="Run a multiline Python script inside the workspace root.",
    ),
)


def build_langchain_tools() -> list[BaseTool]:
    return [_build_langchain_tool(spec) for spec in LANGCHAIN_TOOL_SPECS]


def build_tool_context(config: AgentConfig) -> AgentToolContext:
    return AgentToolContext(
        workspace_root=str(config.workspace_root),
        ignore_patterns=list(config.ignore_patterns),
        shell_timeout_seconds=config.shell_timeout_seconds,
        allowed_shell_commands=list(config.allowed_shell_commands),
        test_command=config.test_command,
    )


def _get_context(runtime: ToolRuntime[AgentToolContext] | None) -> AgentToolContext:
    if runtime is None or runtime.context is None:
        raise ValueError("Tool runtime context is required.")
    return runtime.context


def _get_repository(runtime: ToolRuntime[AgentToolContext] | None) -> Repository:
    context = _get_context(runtime)
    return Repository(_workspace_root(context), context.ignore_patterns)


def _workspace_root(context: AgentToolContext) -> Path:
    return Path(context.workspace_root).expanduser().resolve()


def _build_action_runtime(runtime: ToolRuntime[AgentToolContext] | None) -> ActionRuntime:
    context = _get_context(runtime)
    return ActionRuntime(
        repository=_get_repository(runtime),
        workspace_root=_workspace_root(context),
        shell_timeout_seconds=context.shell_timeout_seconds,
        allowed_shell_commands=list(context.allowed_shell_commands),
        test_command=context.test_command,
    )


def _execute_tool_action(
    action: str,
    runtime: ToolRuntime[AgentToolContext] | None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = execute_action(action, _build_action_runtime(runtime), arguments or {})
    return _result_payload(result)


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
    }


def _build_langchain_tool(spec: LangChainToolSpec) -> BaseTool:
    @tool(spec.name, args_schema=spec.args_schema, description=spec.description)
    def _langchain_tool(
        runtime: ToolRuntime[AgentToolContext] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _execute_tool_action(spec.name, runtime, _clean_tool_arguments(kwargs))

    _langchain_tool.func.__name__ = f"{spec.name}_langchain_tool"
    return _langchain_tool


def _clean_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in arguments.items() if value is not None}
