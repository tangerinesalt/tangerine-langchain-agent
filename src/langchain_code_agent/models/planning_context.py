from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PlanningFileExcerpt:
    path: str
    content: str
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FixFailingTestsPlanningContext:
    test_command: str
    run_tests_ok: bool
    run_tests_returncode: int | None = None
    run_tests_error: str | None = None
    failing_test_ids: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)
    file_excerpts: list[PlanningFileExcerpt] = field(default_factory=list)
    workspace_summary: list[str] = field(default_factory=list)
    failure_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanningContext:
    fix_failing_tests: FixFailingTestsPlanningContext | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.fix_failing_tests is not None:
            payload["fix_failing_tests"] = self.fix_failing_tests.to_dict()
        return payload
