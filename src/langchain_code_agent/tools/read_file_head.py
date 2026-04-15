from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository, RepositoryError


def read_file_head_tool(
    repository: Repository,
    *,
    path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> ToolResult:
    try:
        result = repository.read_text_head(path, start_line=start_line, max_lines=max_lines)
    except RepositoryError as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, data=result)
