from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def write_file_tool(
    repository: Repository,
    *,
    path: str,
    content: str,
    overwrite: bool = False,
) -> ToolResult:
    try:
        result = repository.write_text(path, content, overwrite=overwrite)
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data=result)
