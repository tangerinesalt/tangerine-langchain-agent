from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def insert_text_tool(
    repository: Repository,
    *,
    path: str,
    anchor: str,
    text: str,
    position: str = "after",
) -> ToolResult:
    try:
        result = repository.insert_text(
            path,
            anchor=anchor,
            text=text,
            position=position,
        )
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data=result)
