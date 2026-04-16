from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from langchain_code_agent.actions import action_names


class CompletionCheck(BaseModel):
    check_type: Literal[
        "file_exists",
        "file_absent",
        "file_changed",
        "action_succeeded",
        "shell_output_contains",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PlanStep(BaseModel):
    action: str
    description: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in action_names():
            raise ValueError(f"Unsupported action: {value}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Plan(BaseModel):
    summary: str = Field(min_length=1)
    steps: list[PlanStep] = Field(default_factory=list)
    completion_checks: list[CompletionCheck] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
