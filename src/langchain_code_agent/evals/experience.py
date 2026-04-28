from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from langchain_code_agent.evals.models import EvalCase, EvalCaseResult, EvalReport
from langchain_code_agent.evals.runner import load_eval_case, run_eval_suite

ExperienceOutcome = Literal["success", "expected_failure", "expectation_mismatch"]


class ExperienceRepairRecord(BaseModel):
    problem_code: str | None = None
    repair_code: str
    reason: str | None = None
    agent_success_after_repair: bool
    attempts_after_repair: int
    actions_after_repair: list[str] = Field(default_factory=list)


class ExperienceRecord(BaseModel):
    schema_version: str = "agent-experience-record-v1"
    id: str
    source: Literal["eval"] = "eval"
    source_report_id: str
    case_id: str
    task: str
    workspace_fixture: str
    run_id: str
    artifact_path: str | None = None
    outcome: ExperienceOutcome
    passed_expectations: bool
    agent_success: bool
    expected_success: bool
    actions: list[str] = Field(default_factory=list)
    expected_actions: list[str] = Field(default_factory=list)
    attempts: int
    replanned: bool
    observed_failure_type: str | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    error_types: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    repair: ExperienceRepairRecord | None = None
    duration_ms: int


class ExperienceIndex(BaseModel):
    schema_version: str = "agent-experience-index-v1"
    source_report_id: str
    source_report_schema_version: str
    record_count: int
    by_outcome: dict[str, list[str]] = Field(default_factory=dict)
    by_case_id: dict[str, list[str]] = Field(default_factory=dict)
    by_action: dict[str, list[str]] = Field(default_factory=dict)
    by_failure_type: dict[str, list[str]] = Field(default_factory=dict)
    by_failure_code: dict[str, list[str]] = Field(default_factory=dict)
    by_repair_code: dict[str, list[str]] = Field(default_factory=dict)


class ExperienceArchivePaths(BaseModel):
    records_path: str
    index_path: str


class ExperienceArchive(BaseModel):
    report: EvalReport
    records: list[ExperienceRecord]
    index: ExperienceIndex
    paths: ExperienceArchivePaths | None = None


def build_experience_records(
    cases: Iterable[EvalCase],
    report: EvalReport,
) -> list[ExperienceRecord]:
    cases_by_id = {case.id: case for case in cases}
    records: list[ExperienceRecord] = []
    for result in report.case_results:
        case = cases_by_id.get(result.id)
        if case is None:
            raise ValueError(f"Missing eval case for result id: {result.id}")
        records.append(_build_record(case, result, report))
    return records


def build_experience_index(
    records: Iterable[ExperienceRecord],
    *,
    report: EvalReport,
) -> ExperienceIndex:
    record_list = list(records)
    index = ExperienceIndex(
        source_report_id=report.eval_id,
        source_report_schema_version=report.schema_version,
        record_count=len(record_list),
    )
    for record in record_list:
        _add_index_ref(index.by_outcome, record.outcome, record.id)
        _add_index_ref(index.by_case_id, record.case_id, record.id)
        for action in record.actions:
            _add_index_ref(index.by_action, action, record.id)
        if record.observed_failure_type is not None:
            _add_index_ref(index.by_failure_type, record.observed_failure_type, record.id)
        if record.failure_code is not None:
            _add_index_ref(index.by_failure_code, record.failure_code, record.id)
        if record.repair is not None:
            _add_index_ref(index.by_repair_code, record.repair.repair_code, record.id)
    return index


def write_experience_archive(
    records: Iterable[ExperienceRecord],
    index: ExperienceIndex,
    *,
    archive_dir: str | Path,
) -> ExperienceArchivePaths:
    target = Path(archive_dir)
    target.mkdir(parents=True, exist_ok=True)
    records_path = target / "records.jsonl"
    index_path = target / "index.json"
    record_list = list(records)
    records_payload = "\n".join(record.model_dump_json() for record in record_list)
    if records_payload:
        records_payload += "\n"
    records_path.write_text(records_payload, encoding="utf-8")
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return ExperienceArchivePaths(
        records_path=str(records_path),
        index_path=str(index_path),
    )


def load_experience_records(path: str | Path) -> list[ExperienceRecord]:
    payload = Path(path).read_text(encoding="utf-8")
    return [
        ExperienceRecord.model_validate_json(line)
        for line in payload.splitlines()
        if line.strip()
    ]


def query_experience_records(
    records: Iterable[ExperienceRecord],
    *,
    case_id: str | None = None,
    outcome: ExperienceOutcome | None = None,
    action: str | None = None,
    failure_type: str | None = None,
    failure_code: str | None = None,
    repair_code: str | None = None,
) -> list[ExperienceRecord]:
    matches: list[ExperienceRecord] = []
    for record in records:
        if case_id is not None and record.case_id != case_id:
            continue
        if outcome is not None and record.outcome != outcome:
            continue
        if action is not None and action not in record.actions:
            continue
        if failure_type is not None and record.observed_failure_type != failure_type:
            continue
        if failure_code is not None and record.failure_code != failure_code:
            continue
        if repair_code is not None and (
            record.repair is None or record.repair.repair_code != repair_code
        ):
            continue
        matches.append(record)
    return matches


def archive_eval_suite(
    case_paths: list[str | Path],
    *,
    project_root: Path,
    workspaces_root: Path,
    archive_dir: str | Path,
    report_path: str | Path | None = None,
) -> ExperienceArchive:
    cases = [load_eval_case(path) for path in case_paths]
    report = run_eval_suite(
        case_paths,
        project_root=project_root,
        workspaces_root=workspaces_root,
        report_path=report_path,
    )
    records = build_experience_records(cases, report)
    index = build_experience_index(records, report=report)
    paths = write_experience_archive(records, index, archive_dir=archive_dir)
    return ExperienceArchive(report=report, records=records, index=index, paths=paths)


def _build_record(
    case: EvalCase,
    result: EvalCaseResult,
    report: EvalReport,
) -> ExperienceRecord:
    return ExperienceRecord(
        id=f"{case.id}:{result.run_id}",
        source_report_id=report.eval_id,
        case_id=case.id,
        task=case.task,
        workspace_fixture=case.workspace_fixture,
        run_id=result.run_id,
        artifact_path=result.artifact_path,
        outcome=_classify_outcome(result),
        passed_expectations=result.passed,
        agent_success=result.agent_success,
        expected_success=case.expected_success,
        actions=list(result.actions),
        expected_actions=list(case.expected_actions),
        attempts=result.attempts,
        replanned=result.replanned,
        observed_failure_type=result.observed_failure_type,
        failure_stage=result.failure_stage,
        failure_code=result.failure_code,
        error_types=list(result.error_types),
        event_types=list(result.event_types),
        repair=_build_repair_record(result),
        duration_ms=result.duration_ms,
    )


def _classify_outcome(result: EvalCaseResult) -> ExperienceOutcome:
    if not result.passed:
        return "expectation_mismatch"
    if result.agent_success:
        return "success"
    return "expected_failure"


def _build_repair_record(result: EvalCaseResult) -> ExperienceRepairRecord | None:
    if result.repair_code is None:
        return None
    repair_event = _first_repair_event(result.artifact_path)
    return ExperienceRepairRecord(
        problem_code=_repair_event_value(repair_event, "repaired_failure_code"),
        repair_code=result.repair_code,
        reason=_repair_event_value(repair_event, "reason"),
        agent_success_after_repair=result.agent_success,
        attempts_after_repair=result.attempts,
        actions_after_repair=list(result.actions),
    )


def _first_repair_event(artifact_path: str | None) -> dict[str, Any] | None:
    if artifact_path is None:
        return None
    path = Path(artifact_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") == "plan_repaired":
            return event
    return None


def _repair_event_value(event: dict[str, Any] | None, key: str) -> str | None:
    if event is None:
        return None
    details = event.get("details")
    if not isinstance(details, dict):
        return None
    value = details.get(key)
    if value is None:
        return None
    return str(value)


def _add_index_ref(index: dict[str, list[str]], key: str, record_id: str) -> None:
    refs = index.setdefault(key, [])
    refs.append(record_id)
