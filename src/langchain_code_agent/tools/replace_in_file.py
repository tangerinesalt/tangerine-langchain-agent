from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def replace_in_file_tool(
    repository: Repository,
    *,
    path: str,
    old_text: str,
    new_text: str,
    count: int = 1,
) -> ToolResult:
    try:
        result = repository.replace_in_file(
            path,
            old_text=old_text,
            new_text=new_text,
            count=count,
        )
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data=result)
