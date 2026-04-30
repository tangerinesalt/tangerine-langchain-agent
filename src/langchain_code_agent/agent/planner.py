from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_code_agent.actions import action_argument_schemas_text, action_names_csv
from langchain_code_agent.agent.plan_output_normalizer import normalize_plan_output
from langchain_code_agent.agent.plan_validator import validate_plan
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.llm.factory import build_chat_model
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.planning_context import FixFailingTestsPlanningContext
from langchain_code_agent.models.replan import ReplanContext
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

    def revise_plan(
        self,
        task: Task,
        *,
        invalid_plan: Plan,
        failure_code: str,
        failure_message: str,
    ) -> Plan:
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

    def revise_plan(
        self,
        task: Task,
        *,
        invalid_plan: Plan,
        failure_code: str,
        failure_message: str,
    ) -> Plan:
        return self.create_plan(task)


class LangChainPlanner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def create_plan(self, task: Task) -> Plan:
        request_content = _build_task_request_content(task)
        return self._create_plan(request_content, task)

    def revise_plan(
        self,
        task: Task,
        *,
        invalid_plan: Plan,
        failure_code: str,
        failure_message: str,
    ) -> Plan:
        request_content = _build_plan_revision_request_content(
            task,
            invalid_plan=invalid_plan,
            failure_code=failure_code,
            failure_message=failure_message,
        )
        return self._create_plan(request_content, task)

    def _create_plan(self, request_content: str, task: Task) -> Plan:
        response_mode = self.config.planner_response_mode
        if response_mode not in {"auto", "structured", "json_text"}:
            raise ValueError(f"Unsupported planner_response_mode: {response_mode}")
        if response_mode == "json_text" or _should_use_json_planner_fallback(self.config):
            return self._create_plan_with_json_fallback(request_content, task)
        if response_mode == "structured":
            return self._create_structured_plan(request_content, task)

        try:
            return self._create_structured_plan(request_content, task)
        except Exception as exc:
            if _should_retry_structured_plan_as_json(exc):
                return self._create_plan_with_json_fallback(request_content, task)
            raise

    def _create_structured_plan(self, request_content: str, task: Task) -> Plan:
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
                        "content": request_content,
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

    def _create_plan_with_json_fallback(self, request_content: str, task: Task) -> Plan:
        model = build_chat_model(self.config)
        messages = _build_json_fallback_messages(request_content)
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


def _should_retry_structured_plan_as_json(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = (
        "tool_choice",
        "no endpoints found",
        "response_format",
        "structured output",
        "structured response",
        "function calling",
        "tool calling",
    )
    return any(marker in message for marker in retry_markers)


def _build_json_fallback_messages(request_content: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"{request_content}\n"
                "Return only valid JSON.\n"
                "Available actions and arguments:\n"
                f"{PLANNER_ACTION_SCHEMAS}\n"
                "If the goal is to create a new text file, include a write_file step.\n"
                "If the task depends on the current date or relative dates, include "
                "get_current_date before time-sensitive work.\n"
                "If external information is required and no dedicated tool exists, "
                "prefer run_python_script over a multiline shell string.\n"
                "Never call read_file, read_file_head, replace_in_file, insert_text, "
                "or glob_files with an empty path or empty pattern.\n"
                "For glob_files, use one pattern string per step. Do not use multiple "
                "or patterns arguments.\n"
                "For replace_in_file, use old_text and new_text. Do not use old_str "
                "or new_str.\n"
                "When Planning context JSON is provided, treat it as deterministic "
                "local evidence gathered before planning. Use it to choose relevant "
                "files and avoid restarting with inspection-only steps when it already "
                "includes failing test output or candidate paths.\n"
                "For fix-failing-tests tasks, including Chinese tasks that mention "
                "修复 and 测试, read relevant files, edit code, then run_tests.\n"
                "Include completion_checks when the task has a concrete success condition.\n"
                "Use file_contains for exact expected file text, tests_passed for run_tests, "
                "command_exit_code for shell command return codes, and "
                "no_unexpected_file_changes when only specific paths may change.\n"
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
    if task.planning_context is not None:
        lines.append(
            "Planning context JSON:\n"
            + json.dumps(task.planning_context.to_dict(), ensure_ascii=False, indent=2)
        )
        contract = _build_planning_context_contract(task.planning_context.fix_failing_tests)
        if contract:
            lines.append("Plan contract:\n" + "\n".join(contract))
    if task.replan_context is not None:
        lines.append(
            "Replan context JSON:\n"
            + json.dumps(task.replan_context.to_dict(), ensure_ascii=False, indent=2)
        )
        guidance = _build_replan_guidance(task.replan_context)
        if guidance:
            lines.append("Replan guidance:\n" + "\n".join(guidance))
    lines.append("Generate the best execution plan.")
    return "\n".join(lines)


def _build_plan_revision_request_content(
    task: Task,
    *,
    invalid_plan: Plan,
    failure_code: str,
    failure_message: str,
) -> str:
    lines = [_build_task_request_content(task)]
    lines.append(
        "Previous invalid plan JSON:\n"
        + json.dumps(invalid_plan.to_dict(), ensure_ascii=False, indent=2)
    )
    lines.append(
        "Structural correction request:\n"
        f"- Validation failure code: {failure_code}\n"
        f"- Validation failure message: {failure_message}\n"
        "- Return a complete replacement plan, not a partial diff.\n"
        "- Preserve the task goal and use only allowed actions.\n"
        "- Satisfy the validation failure exactly."
    )
    return "\n".join(lines)


def _existing_workspace_paths(config: AgentConfig) -> set[str]:
    repository = Repository(config.workspace_root, config.ignore_patterns)
    return set(repository.snapshot_file_state())


def _build_replan_guidance(replan_context: ReplanContext) -> list[str]:
    guidance: list[str] = []
    failure_codes = set(replan_context.attempt_failure_codes)

    if "missing_edit_step" in failure_codes:
        guidance.append(
            "- The previous attempt failed with missing_edit_step. This retry must "
            "include at least one concrete code-editing action before the final "
            "run_tests step. Do not return only inspection or diagnostic steps."
        )
    if "missing_validation_step" in failure_codes:
        guidance.append(
            "- The previous attempt failed with missing_validation_step. This retry "
            "must finish with a final run_tests verification step after the edits."
        )
    if "validation_before_edit" in failure_codes:
        guidance.append(
            "- The previous attempt failed with validation_before_edit. Any run_tests "
            "verification step must come after the edit steps, not before them."
        )

    return guidance


def _build_planning_context_contract(
    fix_context: FixFailingTestsPlanningContext | None,
) -> list[str]:
    if fix_context is None:
        return []

    contract: list[str] = [
        "- This is a fix-failing-tests task with deterministic local context already "
        "collected. Do not start again with list_files when candidate_paths are already provided.",
        "- The plan must include at least one concrete code-editing action "
        "(replace_in_file, insert_text, or write_file) targeting a candidate path.",
        "- The final step must be run_tests after the edit steps.",
    ]
    if fix_context.file_excerpts:
        contract.insert(
            1,
            "- Use file_excerpts and candidate_paths to choose the edit target. Add read_file "
            "or read_file_head only when the excerpt is not enough for a concrete edit.",
        )
    else:
        contract.insert(
            1,
            "- Use candidate_paths to choose the edit target and read the relevant file "
            "before making a concrete edit.",
        )

    if fix_context.failing_test_ids:
        contract.append(
            "- Failing tests already identified: "
            + ", ".join(fix_context.failing_test_ids[:4])
            + "."
        )
    if fix_context.candidate_paths:
        contract.append(
            "- Candidate paths to inspect first: "
            + ", ".join(fix_context.candidate_paths[:6])
            + "."
        )
    implementation_candidates = [
        path for path in fix_context.candidate_paths if not _looks_like_test_path(path)
    ]
    if implementation_candidates:
        contract.append(
            "- Prefer editing a non-test candidate path when possible: "
            + ", ".join(implementation_candidates[:4])
            + "."
        )

    return contract


def _looks_like_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    filename = Path(normalized).name
    return filename.startswith("test_") or "/tests/" in f"/{normalized}/"
