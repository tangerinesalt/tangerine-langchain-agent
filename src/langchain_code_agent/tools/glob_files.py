from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository


def glob_files_tool(repository: Repository, *, pattern: str, limit: int = 200) -> ToolResult:
    matches = repository.glob_files(pattern, limit=limit)
    return ToolResult(ok=True, data={"pattern": pattern, "files": matches, "count": len(matches)})
