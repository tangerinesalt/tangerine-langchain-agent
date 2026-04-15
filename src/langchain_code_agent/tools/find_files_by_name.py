from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository


def find_files_by_name_tool(repository: Repository, *, name: str, limit: int = 200) -> ToolResult:
    matches = repository.find_files_by_name(name, limit=limit)
    return ToolResult(ok=True, data={"name": name, "files": matches, "count": len(matches)})
