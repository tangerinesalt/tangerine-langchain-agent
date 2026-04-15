from __future__ import annotations

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository


def search_text_tool(
    repository: Repository,
    *,
    query: str,
    max_results: int = 20,
    case_sensitive: bool = False,
    use_regex: bool = False,
    path_glob: str | None = None,
) -> ToolResult:
    matches = repository.search_text_advanced(
        query,
        max_results=max_results,
        case_sensitive=case_sensitive,
        use_regex=use_regex,
        path_glob=path_glob,
    )
    return ToolResult(
        ok=True,
        data={
            "query": query,
            "matches": matches,
            "count": len(matches),
            "case_sensitive": case_sensitive,
            "use_regex": use_regex,
            "path_glob": path_glob,
        },
    )
