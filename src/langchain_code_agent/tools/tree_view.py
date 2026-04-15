from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def tree_view_tool(repository: Repository, *, path: str = ".", depth: int = 2) -> ToolResult:
    try:
        lines = repository.tree_view(path, depth=depth)
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data={"path": path, "depth": depth, "lines": lines})
