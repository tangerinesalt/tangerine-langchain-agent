from __future__ import annotations

import argparse
import json
from typing import Any, cast

from langchain_code_agent.agent.runner import AgentRunner
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent for a task.")
    run_parser.add_argument("--task", required=True, help="Task goal for the agent.")
    run_parser.add_argument("--workspace", help="Workspace root to inspect.")
    run_parser.add_argument("--config", help="Path to a TOML config file.")
    run_parser.add_argument("--planner", choices=["noop", "langchain"], help="Planner override.")
    run_parser.add_argument(
        "--mode",
        choices=["dry-run", "execute"],
        default="dry-run",
        help="Whether to only plan or to execute plan steps.",
    )
    run_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    doctor_parser = subparsers.add_parser("doctor", help="Print resolved configuration.")
    doctor_parser.add_argument("--workspace", help="Workspace root to inspect.")
    doctor_parser.add_argument("--config", help="Path to a TOML config file.")
    doctor_parser.add_argument("--planner", choices=["noop", "langchain"], help="Planner override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AgentConfig.from_sources(
        workspace_root=args.workspace,
        config_path=args.config,
        planner_backend=args.planner,
    )
    configure_logging(config.log_level)

    if args.command == "doctor":
        print(json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False))
        return 0

    runner = AgentRunner(config)
    result = runner.run(args.task, execution_mode=args.mode)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_human_output(result.to_dict()))
    return 0


def _config_to_dict(config: AgentConfig) -> dict[str, object]:
    return {
        "workspace_root": str(config.workspace_root),
        "planner_backend": config.planner_backend,
        "model_profile": config.model_profile,
        "model_backend": config.model_backend,
        "model_provider": config.model_provider,
        "model": config.model,
        "model_base_url": config.model_base_url,
        "model_timeout_seconds": config.model_timeout_seconds,
        "model_sources": {
            "model_config_path": str(config.model_config_path) if config.model_config_path else None,
            "auth_path": str(config.auth_path) if config.auth_path else None,
            "model_api_key_source": config.model_api_key_source,
        },
        "shell_timeout_seconds": config.shell_timeout_seconds,
        "test_command": config.test_command,
        "log_level": config.log_level,
        "ignore_patterns": config.ignore_patterns,
        "allowed_shell_commands": config.allowed_shell_commands,
    }


def _format_human_output(payload: dict[str, Any]) -> str:
    plan = cast(dict[str, Any], payload["plan"])
    step_results = cast(list[dict[str, Any]], payload["step_results"])
    lines = [
        f"Task: {payload['task']}",
        f"Workspace: {payload['workspace_root']}",
        f"Execution mode: {payload['execution_mode']}",
        f"Planner: {payload['planner']}",
        f"Plan: {plan['summary']}",
    ]
    for index, step in enumerate(plan["steps"], start=1):
        lines.append(f"{index}. {step['action']} - {step['description']}")
    lines.append("Results:")
    for result in step_results:
        status = str(result["status"]).upper()
        lines.append(f"- {result['action']}: {status}")
        if result["error"]:
            lines.append(f"  error: {result['error']}")
    return "\n".join(lines)
