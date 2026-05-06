from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

from pydantic import BaseModel

from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.workspace.repository import Repository

ActionExecutor = Callable[["ActionRuntime", dict[str, object]], ToolResult]


@dataclass(slots=True)
class ActionRuntime:
    repository: Repository
    workspace_root: Path
    shell_timeout_seconds: int
    allowed_shell_commands: list[str]
    test_command: str | None = None


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    executor: ActionExecutor
    produces_shell_output: bool = False
    langchain_description: str | None = None
    langchain_args_schema: type[BaseModel] | None = None

    @property
    def allowed_arguments(self) -> frozenset[str]:
        if self.langchain_args_schema is None:
            return frozenset()
        return frozenset(self.langchain_args_schema.model_fields)

    @property
    def required_arguments(self) -> frozenset[str]:
        if self.langchain_args_schema is None:
            return frozenset()
        return frozenset(
            name
            for name, field_info in self.langchain_args_schema.model_fields.items()
            if field_info.is_required()
        )

    @property
    def planner_arguments_schema(self) -> str:
        if self.langchain_args_schema is None:
            return "{}"
        fields = self.langchain_args_schema.model_fields
        if not fields:
            return "{}"
        parts = [
            f'"{name}": {_field_type_label(field_info.annotation)} '
            f'{"required" if field_info.is_required() else "optional"}'
            for name, field_info in fields.items()
        ]
        return "{" + ", ".join(parts) + "}"


def _field_type_label(annotation: object) -> str:
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is bool:
        return "boolean"
    if origin is list:
        item_args = get_args(annotation)
        if item_args and item_args[0] is str:
            return "string array"
        return "array"
    return "value"


def _strip_optional(annotation: object) -> object:
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation

