from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_code_agent.agent.runner import AgentRunner
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.evals.models import EvalCase, EvalCaseResult, EvalReport
from langchain_code_agent.models.plan import Plan
from langchain_code_agent.models.result import RunResult
from langchain_code_agent.models.task import Task


def load_eval_case(path: str | Path) -> EvalCase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalCase.model_validate(data)


def run_eval_case(
    case: EvalCase,
    *,
    project_root: Path,
    workspaces_root: Path,
) -> EvalCaseResult:
    started = time.perf_counter()
    workspace = _prepare_workspace(
        case,
        project_root=project_root,
        workspaces_root=workspaces_root,
    )
    config = AgentConfig(
        workspace_root=workspace,
        planner_backend=case.planner_backend,
        shell_timeout_seconds=case.shell_timeout_seconds,
        max_replans=case.max_replans,
        test_command=case.test_command,
        ignore_patterns=list(case.ignore_patterns),
        allowed_shell_commands=list(case.allowed_shell_commands),
    )
    runner = AgentRunner(config)
    if case.plans:
        runner.planner = _SequentialPlanner(case.plans)

    result = runner.run(case.task, execution_mode=case.execution_mode)
    duration_ms = int((time.perf_counter() - started) * 1000)
    return _evaluate_case_result(case, result, workspace=workspace, duration_ms=duration_ms)


def run_eval_suite(
    case_paths: list[str | Path],
    *,
    project_root: Path,
    workspaces_root: Path,
    report_path: str | Path | None = None,
) -> EvalReport:
    started_at = datetime.now(UTC).isoformat()
    results = [
        run_eval_case(
            load_eval_case(path),
            project_root=project_root,
            workspaces_root=workspaces_root,
        )
        for path in case_paths
    ]
    report = _build_report(started_at=started_at, case_results=results)
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


class _SequentialPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = list(plans)
        self.calls = 0

    def create_plan(self, task: Task) -> Plan:
        del task
        if self.calls >= len(self.plans):
            return self.plans[-1]
        plan = self.plans[self.calls]
        self.calls += 1
        return plan


def _prepare_workspace(
    case: EvalCase,
    *,
    project_root: Path,
    workspaces_root: Path,
) -> Path:
    source = (project_root / case.workspace_fixture).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Eval fixture does not exist: {source}")
    target = workspaces_root / f"{case.id}-{uuid4().hex[:8]}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def _evaluate_case_result(
    case: EvalCase,
    result: RunResult,
    *,
    workspace: Path,
    duration_ms: int,
) -> EvalCaseResult:
    actions = [step.action for step in result.step_results]
    event_types = [event.event_type for event in result.events]
    error_types = _collect_error_types(result)
    failure_code = _first_failure_code(result)
    repair_code = _first_repair_code(result)
    planning_repaired = repair_code is not None
    attempts = result.final_report.attempts
    failure_reasons: list[str] = []

    if result.final_report.success is not case.expected_success:
        failure_reasons.append(
            "Expected final_report.success "
            f"{case.expected_success}, got {result.final_report.success}."
        )
    if case.expected_actions and actions != case.expected_actions:
        failure_reasons.append(f"Expected actions {case.expected_actions}, got {actions}.")
    if case.expected_attempts is not None and attempts != case.expected_attempts:
        failure_reasons.append(f"Expected attempts {case.expected_attempts}, got {attempts}.")
    if case.expected_failure_code is not None and failure_code != case.expected_failure_code:
        failure_reasons.append(
            f"Expected failure code {case.expected_failure_code}, got {failure_code}."
        )
    if case.expected_repaired is not None and planning_repaired is not case.expected_repaired:
        failure_reasons.append(
            f"Expected planning_repaired {case.expected_repaired}, got {planning_repaired}."
        )
    if case.expected_repair_code is not None and repair_code != case.expected_repair_code:
        failure_reasons.append(
            f"Expected repair code {case.expected_repair_code}, got {repair_code}."
        )

    _check_expected_file_changes(case, result, failure_reasons)
    _check_expected_files(case, workspace, failure_reasons)
    _check_expected_items(
        "error types",
        case.expected_error_types,
        error_types,
        failure_reasons,
    )
    _check_expected_items(
        "event types",
        case.expected_event_types,
        event_types,
        failure_reasons,
    )

    return EvalCaseResult(
        id=case.id,
        run_id=result.run_id,
        artifact_path=result.artifact_path,
        passed=not failure_reasons,
        agent_success=result.final_report.success,
        failure_reasons=failure_reasons,
        failure_type=_classify_failure(error_types, event_types, failure_reasons),
        observed_failure_type=_classify_observed_failure(error_types, event_types),
        failure_stage=_first_failure_stage(result),
        failure_code=failure_code,
        planning_repaired=planning_repaired,
        repair_code=repair_code,
        steps=len(result.step_results),
        tool_calls=len(result.final_report.tool_calls),
        attempts=attempts,
        replanned="replan_requested" in event_types or attempts > 1,
        duration_ms=duration_ms,
        actions=actions,
        error_types=error_types,
        event_types=event_types,
    )


def _check_expected_file_changes(
    case: EvalCase,
    result: RunResult,
    failure_reasons: list[str],
) -> None:
    actual_changes = result.final_report.file_changes
    for expected in case.expected_file_changes:
        if not any(
            change.path == expected.path
            and (expected.change_type is None or change.change_type == expected.change_type)
            for change in actual_changes
        ):
            failure_reasons.append(
                "Missing expected file change "
                f"{expected.path}:{expected.change_type or '*'}."
            )


def _check_expected_files(
    case: EvalCase,
    workspace: Path,
    failure_reasons: list[str],
) -> None:
    for expected in case.expected_files:
        path = workspace / expected.path
        if path.exists() is not expected.exists:
            failure_reasons.append(
                f"Expected file exists={expected.exists} for {expected.path}."
            )
            continue
        if expected.exists and expected.content is not None:
            content = path.read_text(encoding="utf-8")
            if content != expected.content:
                failure_reasons.append(f"Unexpected content for {expected.path}.")


def _check_expected_items(
    label: str,
    expected: list[str],
    actual: list[str],
    failure_reasons: list[str],
) -> None:
    missing = [item for item in expected if item not in actual]
    if missing:
        failure_reasons.append(f"Missing expected {label}: {missing}.")


def _collect_error_types(result: RunResult) -> list[str]:
    seen: set[str] = set()
    error_types: list[str] = []
    errors = list(result.final_report.errors)
    for attempt in result.attempts:
        errors.extend(attempt.errors)
        errors.extend(attempt.completion_errors)
    for error in errors:
        if error.error_type in seen:
            continue
        seen.add(error.error_type)
        error_types.append(error.error_type)
    return error_types


def _first_failure_stage(result: RunResult) -> str | None:
    for error in result.final_report.errors:
        if error.stage is not None:
            return error.stage
    for attempt in result.attempts:
        for error in [*attempt.errors, *attempt.completion_errors]:
            if error.stage is not None:
                return error.stage
    return None


def _first_failure_code(result: RunResult) -> str | None:
    for error in result.final_report.errors:
        if error.failure_code is not None:
            return error.failure_code
    for attempt in result.attempts:
        for error in [*attempt.errors, *attempt.completion_errors]:
            if error.failure_code is not None:
                return error.failure_code
    return None


def _first_repair_code(result: RunResult) -> str | None:
    for event in result.events:
        if event.event_type != "plan_repaired":
            continue
        repair_code = event.details.get("repair_code")
        if repair_code is not None:
            return str(repair_code)
    return None


def _build_report(*, started_at: str, case_results: list[EvalCaseResult]) -> EvalReport:
    total_cases = len(case_results)
    passed_cases = sum(1 for result in case_results if result.passed)
    repair_cases = [result for result in case_results if result.planning_repaired]
    return EvalReport(
        eval_id=uuid4().hex,
        started_at=started_at,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        success_rate=_rate(passed_cases, total_cases),
        avg_steps=_average([result.steps for result in case_results]),
        avg_attempts=_average([result.attempts for result in case_results]),
        replan_rate=_rate(
            sum(1 for result in case_results if result.replanned),
            total_cases,
        ),
        tool_error_rate=_rate(
            sum(1 for result in case_results if "ToolExecutionError" in result.error_types),
            total_cases,
        ),
        completion_failure_rate=_rate(
            sum(1 for result in case_results if "IncompleteTaskResult" in result.error_types),
            total_cases,
        ),
        planning_failure_rate=_rate(
            sum(1 for result in case_results if "planning_failed" in result.event_types),
            total_cases,
        ),
        plan_repair_success_rate=_rate(
            sum(1 for result in repair_cases if result.agent_success),
            len(repair_cases),
        ),
        failure_codes=_count_items(
            result.failure_code for result in case_results if result.failure_code
        ),
        planning_failure_codes=_count_items(
            result.failure_code
            for result in case_results
            if result.failure_code and "planning_failed" in result.event_types
        ),
        repair_codes=_count_items(
            result.repair_code for result in case_results if result.repair_code
        ),
        case_results=case_results,
    )


def _classify_failure(
    error_types: list[str],
    event_types: list[str],
    failure_reasons: list[str],
) -> str | None:
    if not failure_reasons:
        return None
    if "ToolExecutionError" in error_types:
        return "tool_error"
    if "IncompleteTaskResult" in error_types:
        return "completion_failure"
    if "planning_failed" in event_types:
        return "planning_failure"
    return "expectation_mismatch"


def _classify_observed_failure(
    error_types: list[str],
    event_types: list[str],
) -> str | None:
    if "ToolExecutionError" in error_types:
        return "tool_error"
    if "IncompleteTaskResult" in error_types:
        return "completion_failure"
    if "planning_failed" in event_types:
        return "planning_failure"
    return None


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _count_items(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return counts
