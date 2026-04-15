from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def move_file_tool(
    repository: Repository,
    *,
    source_path: str,
    destination_path: str,
) -> ToolResult:
    try:
        result = repository.move_file(source_path, destination_path)
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data=result)
