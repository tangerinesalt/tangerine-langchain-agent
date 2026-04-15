from __future__ import annotations

import re
from typing import Protocol
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_code_agent.config import AgentConfig
from langchain_code_agent.llm.factory import build_chat_model
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.task import Task

PLANNER_SYSTEM_PROMPT = """
You are the planning model for a local code engineering agent.
You are not a general chat assistant. Your job is to produce a short, directly executable plan
for the agent.

Requirements:
- Stay within the provided workspace.
- Only use these actions: glob_files, find_files_by_name, tree_view, list_files, read_file,
  read_file_head, search_text, replace_in_file, insert_text, delete_file, move_file, run_command,
  run_python_script, run_shell, run_tests, write_file.
- Prefer reading or searching files before running shell commands.
- Use run_python_script for multiline Python logic.
- Use run_command when argv-based execution is clearer than a shell string.
- Use run_shell only when simpler actions are insufficient.
- Use write_file when the task requires creating or updating a text file in the workspace.
- Keep the plan concise, practical, and low-noise.
- Use short summaries and short step descriptions.
- Do not add commentary, motivation, or duplicate steps.
- Think as a code agent: inspect, verify, execute, then write outputs.
- If the task depends on time, date, "latest", "today", "tomorrow", "next few days", or
  weather ranges, first anchor the task to an exact current date.
- If the exact current date is not already given in the task, add an early run_shell step using
  python to print the local date before planning time-dependent work.
- When you mention dates in the plan, prefer absolute dates over relative phrases.
- Do not output pseudocode, placeholders, or speculative steps.
""".strip()


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
                        "content": (
                            f"Task: {task.goal}\n"
                            f"Workspace: {task.workspace_root}\n"
                            f"Execution mode: {task.execution_mode}\n"
                            "Generate the best execution plan."
                        ),
                    }
                ]
            }
        )
        structured_response = result.get("structured_response")
        if structured_response is None:
            raise ValueError("LangChain planner did not return a structured response.")
        if isinstance(structured_response, Plan):
            return structured_response
        return Plan.model_validate(structured_response)

    def _create_plan_with_json_fallback(self, task: Task) -> Plan:
        model = build_chat_model(self.config)
        response = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {task.goal}\n"
                        f"Workspace: {task.workspace_root}\n"
                        f"Execution mode: {task.execution_mode}\n"
                        "Return only valid JSON.\n"
                        "Available actions and arguments:\n"
                        "- glob_files: {\"pattern\": string required, "
                        "\"limit\": integer optional}\n"
                        "- find_files_by_name: {\"name\": string required, "
                        "\"limit\": integer optional}\n"
                        "- tree_view: {\"path\": string optional, \"depth\": integer optional}\n"
                        "- list_files: {\"limit\": integer optional}\n"
                        "- read_file: {\"path\": string required}\n"
                        "- read_file_head: {\"path\": string required, "
                        "\"start_line\": integer optional, \"max_lines\": integer optional}\n"
                        "- search_text: {\"query\": string required, "
                        "\"max_results\": integer optional, "
                        "\"case_sensitive\": boolean optional, "
                        "\"use_regex\": boolean optional, "
                        "\"path_glob\": string optional}\n"
                        "- replace_in_file: {\"path\": string required, "
                        "\"old_text\": string required, "
                        "\"new_text\": string required, "
                        "\"count\": integer optional}\n"
                        "- insert_text: {\"path\": string required, "
                        "\"anchor\": string required, "
                        "\"text\": string required, "
                        "\"position\": string optional}\n"
                        "- delete_file: {\"path\": string required}\n"
                        "- move_file: {\"source_path\": string required, "
                        "\"destination_path\": string required}\n"
                        "- run_command: {\"argv\": string array required, "
                        "\"working_directory\": string optional}\n"
                        "- run_python_script: {\"script\": string required, "
                        "\"working_directory\": string optional}\n"
                        "- run_shell: {\"command\": string required, "
                        "\"working_directory\": string optional}\n"
                        "- run_tests: {\"working_directory\": string optional}\n"
                        "- write_file: {\"path\": string required, "
                        "\"content\": string required, "
                        "\"overwrite\": boolean optional}\n"
                        "If the goal is to create a new text file, include a write_file step.\n"
                        "If external information is required and no dedicated tool exists, "
                        "prefer run_python_script over a multiline shell string.\n"
                        "Return JSON matching this structure exactly:\n"
                        '{"summary":"string","steps":[{"action":"write_file",'
                        '"description":"string","arguments":{"path":"notes.txt",'
                        '"content":"hello","overwrite":false}}]}'
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = _extract_json_object(content)
        return Plan.model_validate_json(content)


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


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return stripped
    return stripped[start : end + 1]
