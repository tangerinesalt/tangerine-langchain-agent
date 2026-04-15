from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def read_file_tool(repository: Repository, *, path: str) -> ToolResult:
    try:
        content = repository.read_text(path)
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data={"path": path, "content": content})
