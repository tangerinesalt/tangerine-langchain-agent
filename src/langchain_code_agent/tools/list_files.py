from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository


def list_files_tool(repository: Repository, *, limit: int = 200) -> ToolResult:
    files = repository.list_files(limit=limit)
    return ToolResult(ok=True, data={"files": files, "count": len(files)})
