from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from langchain_code_agent.agent.plan_normalization_rules import (
    apply_json_text_repairs,
    apply_plan_normalization_rules,
)
from langchain_code_agent.config import AgentConfig
from langchain_code_agent.models.plan import Plan
from langchain_code_agent.models.task import Task


def normalize_plan_output(
    raw_response: Any,
    *,
    task: Task,
    config: AgentConfig,
    response_mode: str,
    retry_callback: Callable[[], Any] | None = None,
) -> Plan:
    if response_mode == "structured":
        plan = _coerce_structured_plan(raw_response)
    elif response_mode == "json_text":
        plan = _parse_json_plan(raw_response, retry_callback=retry_callback)
    else:
        raise ValueError(f"Unsupported response mode: {response_mode}")
    return _normalize_plan(plan, task=task, workspace_root=config.workspace_root)


def _coerce_structured_plan(raw_response: Any) -> Plan:
    if isinstance(raw_response, Plan):
        return raw_response
    if isinstance(raw_response, dict):
        return Plan.model_validate(raw_response)
    raise ValueError("Planner returned an unsupported structured response.")


def _parse_json_plan(
    raw_response: Any,
    *,
    retry_callback: Callable[[], Any] | None = None,
) -> Plan:
    first_text = _extract_text(raw_response)
    try:
        return _parse_json_text(first_text)
    except ValidationError as first_error:
        if retry_callback is None:
            raise ValueError(f"Planner returned invalid JSON: {first_error}") from first_error

    retry_text = _extract_text(retry_callback())
    try:
        return _parse_json_text(retry_text)
    except ValidationError as second_error:
        raise ValueError(f"Planner returned invalid JSON after retry: {second_error}") from second_error


def _parse_json_text(text: str) -> Plan:
    cleaned = apply_json_text_repairs(text)
    return Plan.model_validate_json(cleaned)


def _extract_text(raw_response: Any) -> str:
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content
    return str(content)
def _normalize_plan(plan: Plan, *, task: Task, workspace_root) -> Plan:
    return apply_plan_normalization_rules(plan, task=task, workspace_root=workspace_root)
