from __future__ import annotations

from pydantic import BaseModel, Field


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
