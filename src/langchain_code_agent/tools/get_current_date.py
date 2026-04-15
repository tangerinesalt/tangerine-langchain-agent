from __future__ import annotations

from datetime import datetime

from langchain_code_agent.tools.base import ToolResult


def get_current_date_tool() -> ToolResult:
    current = datetime.now().astimezone()
    return ToolResult(
        ok=True,
        data={
            "current_date": current.date().isoformat(),
            "current_datetime": current.isoformat(),
            "timezone": str(current.tzinfo),
            "utc_offset": current.strftime("%z"),
            "weekday": current.strftime("%A"),
        },
    )
