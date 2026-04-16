from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from langchain_code_agent.actions import (
    ActionRuntime,
    ActionSpec,
    action_langchain_specs,
    execute_action,
)
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.workspace.repository import Repository


@dataclass(slots=True)
class AgentToolContext:
    workspace_root: str
    ignore_patterns: list[str]
    shell_timeout_seconds: int
    allowed_shell_commands: list[str]
    test_command: str | None = None


def build_langchain_tools() -> list[BaseTool]:
    return [_build_langchain_tool(spec) for spec in action_langchain_specs()]


def build_tool_context(config: AgentConfig) -> AgentToolContext:
    return AgentToolContext(
        workspace_root=str(config.workspace_root),
        ignore_patterns=list(config.ignore_patterns),
        shell_timeout_seconds=config.shell_timeout_seconds,
        allowed_shell_commands=list(config.allowed_shell_commands),
        test_command=config.test_command,
    )


def _get_context(runtime: ToolRuntime[AgentToolContext] | None) -> AgentToolContext:
    if runtime is None or runtime.context is None:
        raise ValueError("Tool runtime context is required.")
    return runtime.context


def _get_repository(runtime: ToolRuntime[AgentToolContext] | None) -> Repository:
    context = _get_context(runtime)
    return Repository(_workspace_root(context), context.ignore_patterns)


def _workspace_root(context: AgentToolContext) -> Path:
    return Path(context.workspace_root).expanduser().resolve()


def _build_action_runtime(runtime: ToolRuntime[AgentToolContext] | None) -> ActionRuntime:
    context = _get_context(runtime)
    return ActionRuntime(
        repository=_get_repository(runtime),
        workspace_root=_workspace_root(context),
        shell_timeout_seconds=context.shell_timeout_seconds,
        allowed_shell_commands=list(context.allowed_shell_commands),
        test_command=context.test_command,
    )


def _execute_tool_action(
    action: str,
    runtime: ToolRuntime[AgentToolContext] | None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = execute_action(action, _build_action_runtime(runtime), arguments or {})
    return _result_payload(result)


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
    }


def _build_langchain_tool(spec: ActionSpec) -> BaseTool:
    @tool(
        spec.name,
        args_schema=spec.langchain_args_schema,
        description=spec.langchain_description,
    )
    def _langchain_tool(
        runtime: ToolRuntime[AgentToolContext] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _execute_tool_action(spec.name, runtime, _clean_tool_arguments(kwargs))

    return cast(BaseTool, _langchain_tool)


def _clean_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in arguments.items() if value is not None}
