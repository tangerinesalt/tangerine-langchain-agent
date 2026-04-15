from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic import field_validator

from langchain_code_agent.actions import action_names


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

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
