from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    action: Literal[
        "glob_files",
        "find_files_by_name",
        "tree_view",
        "read_file_head",
        "list_files",
        "read_file",
        "search_text",
        "replace_in_file",
        "insert_text",
        "delete_file",
        "move_file",
        "run_command",
        "run_python_script",
        "run_shell",
        "run_tests",
        "write_file",
    ]
    description: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Plan(BaseModel):
    summary: str = Field(min_length=1)
    steps: list[PlanStep] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
