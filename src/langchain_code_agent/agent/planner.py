from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_code_agent.actions import action_argument_schemas_text, action_names_csv
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.agent.plan_output_normalizer import normalize_plan_output
from langchain_code_agent.agent.plan_validator import validate_plan
from langchain_code_agent.llm.factory import build_chat_model
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.task import Task
from langchain_code_agent.workspace.repository import Repository

PLANNER_ACTIONS = action_names_csv()
PLANNER_ACTION_SCHEMAS = action_argument_schemas_text()
PROMPT_PATH = Path(__file__).resolve().parents[1] / "llm" / "prompts" / "planner.txt"


def _load_planner_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip().format(
        planner_actions=PLANNER_ACTIONS
    )


PLANNER_SYSTEM_PROMPT = _load_planner_system_prompt()


class Planner(Protocol):
    def create_plan(self, task: Task) -> Plan:
        ...


class NoopPlanner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def create_plan(self, task: Task) -> Plan:
        keywords = _extract_keywords(task.goal)
        steps = [
            PlanStep(
                action="list_files",
                description=(
                    f"Inspect the repository layout for {task.workspace_root.name} before "
                    "taking action."
                ),
                arguments={"limit": 100},
            )
        ]
        if keywords:
            steps.append(
                PlanStep(
                    action="search_text",
                    description="Search for likely relevant code based on task keywords.",
                    arguments={"query": " ".join(keywords[:3]), "max_results": 20},
                )
            )
        if self.config.test_command and _should_run_tests(task.goal):
            steps.append(
                PlanStep(
                    action="run_tests",
                    description="Run the configured test command for quick feedback.",
                    arguments={},
                )
            )
        return Plan(summary="A minimal local execution plan.", steps=steps)


class LangChainPlanner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def create_plan(self, task: Task) -> Plan:
        if _should_use_json_planner_fallback(self.config):
            return self._create_plan_with_json_fallback(task)

        agent = create_agent(
            model=build_chat_model(self.config),
            tools=[],
            system_prompt=PLANNER_SYSTEM_PROMPT,
            response_format=Plan,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _build_task_request_content(task),
                    }
                ]
            }
        )
        raw_response = result.get("structured_response")
        if raw_response is None:
            raise ValueError("LangChain planner did not return a structured response.")
        return validate_plan(
            normalize_plan_output(
                raw_response,
                task=task,
                config=self.config,
                response_mode="structured",
            ),
            existing_paths=_existing_workspace_paths(self.config),
        )

    def _create_plan_with_json_fallback(self, task: Task) -> Plan:
        model = build_chat_model(self.config)
        messages = _build_json_fallback_messages(task)
        response = model.invoke(messages)
        return validate_plan(
            normalize_plan_output(
                response,
                task=task,
                config=self.config,
                response_mode="json_text",
                retry_callback=lambda: model.invoke(messages),
            ),
            existing_paths=_existing_workspace_paths(self.config),
        )


def build_planner(config: AgentConfig) -> Planner:
    if config.planner_backend == "noop":
        return NoopPlanner(config)
    if config.planner_backend == "langchain":
        return LangChainPlanner(config)
    raise ValueError(f"Unsupported planner backend: {config.planner_backend}")


def _extract_keywords(task: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", task.lower())
    stop_words = {"the", "and", "for", "with", "that", "this", "from", "into", "tests"}
    return [token for token in tokens if token not in stop_words]


def _should_run_tests(task: str) -> bool:
    lowered = task.lower()
    return any(keyword in lowered for keyword in ("test", "fix", "bug", "failing"))


def _should_use_json_planner_fallback(config: AgentConfig) -> bool:
    if config.model_backend == "local_http":
        return True
    if config.model_backend != "langchain":
        return False
    if not config.model_base_url:
        return False

    parsed_url = urlparse(config.model_base_url)
    hostname = (parsed_url.hostname or "").lower()
    path = parsed_url.path.rstrip("/")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    return hostname in local_hosts and path == "/v1"


def _build_json_fallback_messages(task: Task) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"{_build_task_request_content(task)}\n"
                "Return only valid JSON.\n"
                "Available actions and arguments:\n"
                f"{PLANNER_ACTION_SCHEMAS}\n"
                "If the goal is to create a new text file, include a write_file step.\n"
                "If the task depends on the current date or relative dates, include "
                "get_current_date before time-sensitive work.\n"
                "If external information is required and no dedicated tool exists, "
                "prefer run_python_script over a multiline shell string.\n"
                "Include completion_checks when the task has a concrete success condition.\n"
                "Return JSON matching this structure exactly:\n"
                '{"summary":"string","steps":[{"action":"write_file",'
                '"description":"string","arguments":{"path":"notes.txt",'
                '"content":"hello","overwrite":false}}],'
                '"completion_checks":[{"check_type":"file_exists",'
                '"arguments":{"path":"notes.txt"}}]}'
            )
        ),
    ]


def _build_task_request_content(task: Task) -> str:
    lines = [
        f"Task: {task.goal}",
        f"Workspace: {task.workspace_root}",
        f"Execution mode: {task.execution_mode}",
    ]
    if task.replan_context is not None:
        lines.append(
            "Replan context JSON:\n"
            + json.dumps(task.replan_context.to_dict(), ensure_ascii=False, indent=2)
        )
    lines.append("Generate the best execution plan.")
    return "\n".join(lines)


def _existing_workspace_paths(config: AgentConfig) -> set[str]:
    repository = Repository(config.workspace_root, config.ignore_patterns)
    return set(repository.snapshot_file_state())
