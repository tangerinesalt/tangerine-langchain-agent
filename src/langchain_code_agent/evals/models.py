from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from langchain_code_agent.models.plan import Plan


class ExpectedFileChange(BaseModel):
    path: str
    change_type: str | None = None


class ExpectedFileState(BaseModel):
    path: str
    exists: bool = True
    content: str | None = None


class EvalCase(BaseModel):
    id: str
    task: str
    workspace_fixture: str
    execution_mode: Literal["dry-run", "execute"] = "execute"
    planner_backend: Literal["noop", "langchain"] = "noop"
    shell_timeout_seconds: int = 10
    max_replans: int = 1
    test_command: str | None = None
    ignore_patterns: list[str] = Field(default_factory=lambda: ["__pycache__"])
    allowed_shell_commands: list[str] = Field(
        default_factory=lambda: ["python", "pytest", "rg", "git"]
    )
    plans: list[Plan] = Field(default_factory=list)
    expected_success: bool
    expected_actions: list[str] = Field(default_factory=list)
    expected_file_changes: list[ExpectedFileChange] = Field(default_factory=list)
    expected_files: list[ExpectedFileState] = Field(default_factory=list)
    expected_error_types: list[str] = Field(default_factory=list)
    expected_event_types: list[str] = Field(default_factory=list)
    expected_failure_code: str | None = None
    expected_repaired: bool | None = None
    expected_repair_code: str | None = None
    expected_attempts: int | None = None


class EvalCaseResult(BaseModel):
    schema_version: str = "eval-case-result-v2"
    id: str
    run_id: str
    artifact_path: str | None = None
    passed: bool
    agent_success: bool
    failure_reasons: list[str] = Field(default_factory=list)
    failure_type: str | None = None
    observed_failure_type: str | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    planning_repaired: bool = False
    repair_code: str | None = None
    steps: int
    tool_calls: int
    attempts: int
    replanned: bool
    duration_ms: int
    actions: list[str] = Field(default_factory=list)
    error_types: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    schema_version: str = "eval-report-v2"
    eval_id: str
    started_at: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    success_rate: float
    avg_steps: float
    avg_attempts: float
    replan_rate: float
    tool_error_rate: float
    completion_failure_rate: float
    planning_failure_rate: float
    plan_repair_success_rate: float
    failure_codes: dict[str, int] = Field(default_factory=dict)
    planning_failure_codes: dict[str, int] = Field(default_factory=dict)
    repair_codes: dict[str, int] = Field(default_factory=dict)
    case_results: list[EvalCaseResult]
