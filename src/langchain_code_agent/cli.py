from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from langchain_code_agent.agent.runner import AgentRunner
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.evals.experience import archive_eval_suite
from langchain_code_agent.evals.models import EvalReport
from langchain_code_agent.evals.runner import run_eval_suite
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

    eval_parser = subparsers.add_parser("eval", help="Run agent eval harness tasks.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_run_parser = eval_subparsers.add_parser("run", help="Run eval cases.")
    _add_eval_common_arguments(
        eval_run_parser,
        default_workspaces=".lca/evals/workspaces",
        default_report=".lca/evals/latest.json",
    )
    eval_run_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    eval_archive_parser = eval_subparsers.add_parser(
        "archive",
        help="Run eval cases and write an experience archive.",
    )
    _add_eval_common_arguments(
        eval_archive_parser,
        default_workspaces=".lca/evals/experience-workspaces",
        default_report=".lca/evals/experience-report.json",
    )
    eval_archive_parser.add_argument(
        "--archive-dir",
        default=".lca/evals/experience",
        help="Directory for records.jsonl and index.json.",
    )
    eval_archive_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return _handle_eval_command(args)

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
            "model_config_path": (
                str(config.model_config_path) if config.model_config_path else None
            ),
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


def _add_eval_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_workspaces: str,
    default_report: str,
) -> None:
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve cases and workspace fixtures.",
    )
    parser.add_argument(
        "--cases",
        default="tests/fixtures/agent_tasks",
        help="Eval case file, directory, or glob relative to --project-root.",
    )
    parser.add_argument(
        "--workspaces",
        default=default_workspaces,
        help="Directory where isolated eval workspaces are created.",
    )
    parser.add_argument(
        "--report",
        default=default_report,
        help="Path where the eval report JSON is written.",
    )
    parser.add_argument(
        "--log-level",
        default="CRITICAL",
        help="Logging level for eval runner diagnostics.",
    )


def _handle_eval_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    project_root = Path(args.project_root).expanduser().resolve()
    case_paths = _resolve_eval_case_paths(args.cases, project_root=project_root)
    workspaces_root = _resolve_path(args.workspaces, project_root=project_root)
    report_path = _resolve_path(args.report, project_root=project_root)

    if args.eval_command == "run":
        report = run_eval_suite(
            case_paths,
            project_root=project_root,
            workspaces_root=workspaces_root,
            report_path=report_path,
        )
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(_format_eval_report(report, report_path=report_path))
        return 0

    if args.eval_command == "archive":
        archive_dir = _resolve_path(args.archive_dir, project_root=project_root)
        archive = archive_eval_suite(
            case_paths,
            project_root=project_root,
            workspaces_root=workspaces_root,
            archive_dir=archive_dir,
            report_path=report_path,
        )
        if args.json:
            print(archive.model_dump_json(indent=2))
        else:
            print(_format_eval_archive(archive.model_dump(mode="json")))
        return 0

    raise ValueError(f"Unsupported eval command: {args.eval_command}")


def _resolve_eval_case_paths(cases: str, *, project_root: Path) -> list[str | Path]:
    raw_path = Path(cases).expanduser()
    target = raw_path if raw_path.is_absolute() else project_root / raw_path
    if target.is_file():
        return [target]
    if target.is_dir():
        paths: list[str | Path] = [path for path in sorted(target.glob("*.json"))]
        if paths:
            return paths

    glob_matches = sorted(project_root.glob(cases))
    paths = [path for path in glob_matches if path.is_file()]
    if paths:
        return paths
    raise FileNotFoundError(f"No eval case JSON files matched: {cases}")


def _resolve_path(path: str, *, project_root: Path) -> Path:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return project_root / raw_path


def _format_eval_report(report: EvalReport, *, report_path: Path) -> str:
    return "\n".join(
        [
            f"Eval report: {report.passed_cases}/{report.total_cases} passed",
            f"Schema: {report.schema_version}",
            f"Success rate: {report.success_rate:.3f}",
            f"Replan rate: {report.replan_rate:.3f}",
            f"Completion failure rate: {report.completion_failure_rate:.3f}",
            f"Planning failure codes: {report.planning_failure_codes}",
            f"Repair codes: {report.repair_codes}",
            f"Report path: {report_path}",
        ]
    )


def _format_eval_archive(payload: dict[str, Any]) -> str:
    report = cast(dict[str, Any], payload["report"])
    index = cast(dict[str, Any], payload["index"])
    paths = cast(dict[str, Any] | None, payload["paths"])
    lines = [
        f"Experience archive: {index['record_count']} records",
        f"Eval report: {report['passed_cases']}/{report['total_cases']} passed",
        f"Index schema: {index['schema_version']}",
        f"Outcomes: {index['by_outcome']}",
        f"Failure codes: {index['by_failure_code']}",
        f"Repair codes: {index['by_repair_code']}",
    ]
    if paths is not None:
        lines.append(f"Records path: {paths['records_path']}")
        lines.append(f"Index path: {paths['index_path']}")
    return "\n".join(lines)
