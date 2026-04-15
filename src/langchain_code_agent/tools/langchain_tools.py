from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_code_agent.config import AgentConfig
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


@tool(
    "get_current_date",
    args_schema=GetCurrentDateInput,
    description="Return the current local date, datetime, timezone, and weekday.",
)
def get_current_date_langchain_tool(
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    _get_context(runtime)
    return _result_payload(get_current_date_tool())


@tool(
    "list_files",
    args_schema=ListFilesInput,
    description="List repository files under the workspace root. Use this first to inspect layout.",
)
def list_files_langchain_tool(
    limit: int = 200,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(list_files_tool(repository, limit=limit))


@tool(
    "glob_files",
    args_schema=GlobFilesInput,
    description="Find files by glob pattern relative to the workspace root.",
)
def glob_files_langchain_tool(
    pattern: str,
    limit: int = 200,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(glob_files_tool(repository, pattern=pattern, limit=limit))


@tool(
    "find_files_by_name",
    args_schema=FindFilesByNameInput,
    description="Find files whose names contain the provided fragment.",
)
def find_files_by_name_langchain_tool(
    name: str,
    limit: int = 200,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(find_files_by_name_tool(repository, name=name, limit=limit))


@tool(
    "tree_view",
    args_schema=TreeViewInput,
    description="Return a compact directory tree view for a path inside the workspace root.",
)
def tree_view_langchain_tool(
    path: str = ".",
    depth: int = 2,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(tree_view_tool(repository, path=path, depth=depth))


@tool(
    "read_file",
    args_schema=ReadFileInput,
    description="Read a text file from the workspace using a relative path.",
)
def read_file_langchain_tool(
    path: str,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(read_file_tool(repository, path=path))


@tool(
    "read_file_head",
    args_schema=ReadFileHeadInput,
    description="Read a range of lines from a text file inside the workspace root.",
)
def read_file_head_langchain_tool(
    path: str,
    start_line: int = 1,
    max_lines: int = 200,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        read_file_head_tool(
            repository,
            path=path,
            start_line=start_line,
            max_lines=max_lines,
        )
    )


@tool(
    "search_text",
    args_schema=SearchTextInput,
    description="Search repository text content and return matching lines with file paths.",
)
def search_text_langchain_tool(
    query: str,
    max_results: int = 20,
    case_sensitive: bool = False,
    use_regex: bool = False,
    path_glob: str | None = None,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        search_text_tool(
            repository,
            query=query,
            max_results=max_results,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            path_glob=path_glob,
        )
    )


@tool(
    "run_shell",
    args_schema=RunShellInput,
    description=(
        "Run a whitelisted shell command within the workspace root. "
        "Use only when file inspection is insufficient."
    ),
)
def run_shell_langchain_tool(
    command: str,
    working_directory: str | None = None,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    context = _get_context(runtime)
    return _result_payload(
        run_shell_tool(
            command=command,
            workspace_root=_workspace_root(context),
            timeout_seconds=context.shell_timeout_seconds,
            allowed_commands=context.allowed_shell_commands,
            working_directory=working_directory,
        )
    )


@tool(
    "run_tests",
    args_schema=RunTestsInput,
    description="Run the configured test command inside the workspace root.",
)
def run_tests_langchain_tool(
    working_directory: str | None = None,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    context = _get_context(runtime)
    return _result_payload(
        run_tests_tool(
            test_command=context.test_command,
            workspace_root=_workspace_root(context),
            timeout_seconds=context.shell_timeout_seconds,
            allowed_commands=context.allowed_shell_commands,
            working_directory=working_directory,
        )
    )


@tool(
    "write_file",
    args_schema=WriteFileInput,
    description="Write a UTF-8 text file inside the workspace root.",
)
def write_file_langchain_tool(
    path: str,
    content: str,
    overwrite: bool = False,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        write_file_tool(
            repository,
            path=path,
            content=content,
            overwrite=overwrite,
        )
    )


@tool(
    "replace_in_file",
    args_schema=ReplaceInFileInput,
    description="Replace text inside an existing workspace file.",
)
def replace_in_file_langchain_tool(
    path: str,
    old_text: str,
    new_text: str,
    count: int = 1,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        replace_in_file_tool(
            repository,
            path=path,
            old_text=old_text,
            new_text=new_text,
            count=count,
        )
    )


@tool(
    "insert_text",
    args_schema=InsertTextInput,
    description="Insert text before or after an anchor inside an existing workspace file.",
)
def insert_text_langchain_tool(
    path: str,
    anchor: str,
    text: str,
    position: str = "after",
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        insert_text_tool(
            repository,
            path=path,
            anchor=anchor,
            text=text,
            position=position,
        )
    )


@tool(
    "delete_file",
    args_schema=DeleteFileInput,
    description="Delete an existing file inside the workspace root.",
)
def delete_file_langchain_tool(
    path: str,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(delete_file_tool(repository, path=path))


@tool(
    "move_file",
    args_schema=MoveFileInput,
    description="Move or rename a file inside the workspace root.",
)
def move_file_langchain_tool(
    source_path: str,
    destination_path: str,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    repository = _get_repository(runtime)
    return _result_payload(
        move_file_tool(
            repository,
            source_path=source_path,
            destination_path=destination_path,
        )
    )


@tool(
    "run_command",
    args_schema=RunCommandInput,
    description="Run a whitelisted command using argv form for safer execution.",
)
def run_command_langchain_tool(
    argv: list[str],
    working_directory: str | None = None,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    context = _get_context(runtime)
    return _result_payload(
        run_command_tool(
            argv=argv,
            workspace_root=_workspace_root(context),
            timeout_seconds=context.shell_timeout_seconds,
            allowed_commands=context.allowed_shell_commands,
            working_directory=working_directory,
        )
    )


@tool(
    "run_python_script",
    args_schema=RunPythonScriptInput,
    description="Run a multiline Python script inside the workspace root.",
)
def run_python_script_langchain_tool(
    script: str,
    working_directory: str | None = None,
    runtime: ToolRuntime[AgentToolContext] | None = None,
) -> dict[str, Any]:
    context = _get_context(runtime)
    return _result_payload(
        run_python_script_tool(
            script=script,
            workspace_root=_workspace_root(context),
            timeout_seconds=context.shell_timeout_seconds,
            allowed_commands=context.allowed_shell_commands,
            working_directory=working_directory,
        )
    )


def build_langchain_tools() -> list[BaseTool]:
    return [
        get_current_date_langchain_tool,
        list_files_langchain_tool,
        glob_files_langchain_tool,
        find_files_by_name_langchain_tool,
        tree_view_langchain_tool,
        read_file_langchain_tool,
        read_file_head_langchain_tool,
        search_text_langchain_tool,
        run_shell_langchain_tool,
        run_tests_langchain_tool,
        write_file_langchain_tool,
        replace_in_file_langchain_tool,
        insert_text_langchain_tool,
        delete_file_langchain_tool,
        move_file_langchain_tool,
        run_command_langchain_tool,
        run_python_script_langchain_tool,
    ]


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


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
    }
