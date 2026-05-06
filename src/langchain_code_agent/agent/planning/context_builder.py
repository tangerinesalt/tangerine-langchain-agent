from __future__ import annotations

import re
from pathlib import PurePosixPath

from langchain_code_agent.actions import ActionRuntime
from langchain_code_agent.agent.planning.validator import is_fix_failing_tests_task
from langchain_code_agent.common.text import excerpt_text
from langchain_code_agent.models.planning_context import (
    FixFailingTestsPlanningContext,
    PlanningContext,
    PlanningFileExcerpt,
)
from langchain_code_agent.tools.run_tests import run_tests_tool
from langchain_code_agent.workspace.repository import Repository

MAX_WORKSPACE_SUMMARY_LINES = 24
MAX_FAILURE_EXCERPT_CHARS = 1600
MAX_FAILING_TEST_IDS = 8
MAX_CANDIDATE_PATHS = 12
MAX_FILE_EXCERPTS = 4
MAX_FILE_EXCERPT_LINES = 200
MAX_FILE_EXCERPT_CHARS = 6000
_PYTHON_PATH_PATTERN = re.compile(r"([A-Za-z0-9_./\\-]+\.py)(?::\d+)?")
_FAILED_TEST_PATTERN = re.compile(r"^FAILED\s+([^\s]+::[^\s]+)", re.MULTILINE)
_FROM_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+",
    re.MULTILINE,
)
_IMPORT_PATTERN = re.compile(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)", re.MULTILINE)


def build_planning_context(
    task_text: str,
    *,
    execution_mode: str,
    repository: Repository,
    action_runtime: ActionRuntime,
) -> PlanningContext | None:
    if execution_mode != "execute":
        return None
    if not is_fix_failing_tests_task(task_text):
        return None
    if not action_runtime.test_command:
        return None

    run_result = run_tests_tool(
        test_command=action_runtime.test_command,
        workspace_root=action_runtime.workspace_root,
        timeout_seconds=action_runtime.shell_timeout_seconds,
        allowed_commands=action_runtime.allowed_shell_commands,
    )
    stdout = str(run_result.data.get("stdout") or "")
    stderr = str(run_result.data.get("stderr") or "")
    combined_output = _combine_output(stdout, stderr)
    candidate_paths = _extract_candidate_paths(combined_output, repository)

    context = FixFailingTestsPlanningContext(
        test_command=action_runtime.test_command,
        run_tests_ok=run_result.ok,
        run_tests_returncode=_coerce_optional_int(run_result.data.get("returncode")),
        run_tests_error=run_result.error,
        failing_test_ids=_extract_failing_test_ids(combined_output),
        candidate_paths=candidate_paths,
        file_excerpts=_build_file_excerpts(candidate_paths, repository),
        workspace_summary=_build_workspace_summary(repository),
        failure_excerpt=_excerpt_output(combined_output),
    )
    return PlanningContext(fix_failing_tests=context)


def _build_workspace_summary(repository: Repository) -> list[str]:
    return repository.tree_view(".", depth=2)[:MAX_WORKSPACE_SUMMARY_LINES]


def _combine_output(stdout: str, stderr: str) -> str:
    parts = [value.strip() for value in (stdout, stderr) if value.strip()]
    return "\n\n".join(parts)


def _excerpt_output(output: str) -> str | None:
    return excerpt_text(output, max_chars=MAX_FAILURE_EXCERPT_CHARS)


def _build_file_excerpts(
    candidate_paths: list[str],
    repository: Repository,
) -> list[PlanningFileExcerpt]:
    excerpts: list[PlanningFileExcerpt] = []
    for path in candidate_paths[:MAX_FILE_EXCERPTS]:
        try:
            content = repository.read_text(path)
        except Exception:
            continue
        excerpt, truncated = _excerpt_file_content(content)
        excerpts.append(
            PlanningFileExcerpt(
                path=path,
                content=excerpt,
                truncated=truncated,
            )
        )
    return excerpts


def _excerpt_file_content(content: str) -> tuple[str, bool]:
    lines = content.splitlines()
    selected = lines[:MAX_FILE_EXCERPT_LINES]
    excerpt = "\n".join(selected)
    truncated = len(lines) > MAX_FILE_EXCERPT_LINES
    if len(excerpt) > MAX_FILE_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_FILE_EXCERPT_CHARS].rstrip()
        truncated = True
    return excerpt, truncated


def _extract_failing_test_ids(output: str) -> list[str]:
    failing_ids: list[str] = []
    for match in _FAILED_TEST_PATTERN.finditer(output):
        failing_id = match.group(1).strip()
        if failing_id in failing_ids:
            continue
        failing_ids.append(failing_id)
        if len(failing_ids) >= MAX_FAILING_TEST_IDS:
            break
    return failing_ids


def _extract_candidate_paths(output: str, repository: Repository) -> list[str]:
    known_paths = set(repository.snapshot_file_state())
    candidate_paths: list[str] = []
    for raw_path in _PYTHON_PATH_PATTERN.findall(output):
        normalized = PurePosixPath(raw_path.replace("\\", "/")).as_posix().lstrip("./")
        if normalized not in known_paths:
            continue
        if normalized in candidate_paths:
            continue
        candidate_paths.append(normalized)
        if len(candidate_paths) >= MAX_CANDIDATE_PATHS:
            return candidate_paths

    for related_path in _extract_related_module_paths(candidate_paths, repository, known_paths):
        if related_path in candidate_paths:
            continue
        candidate_paths.append(related_path)
        if len(candidate_paths) >= MAX_CANDIDATE_PATHS:
            break
    return candidate_paths


def _extract_related_module_paths(
    candidate_paths: list[str],
    repository: Repository,
    known_paths: set[str],
) -> list[str]:
    related_paths: list[str] = []
    for path in candidate_paths:
        if not _looks_like_test_file(path):
            continue
        try:
            content = repository.read_text(path)
        except Exception:
            continue
        for module_name in _extract_imported_modules(content):
            module_path = module_name.replace(".", "/") + ".py"
            if module_path not in known_paths:
                continue
            if module_path in related_paths:
                continue
            related_paths.append(module_path)
    return related_paths


def _extract_imported_modules(content: str) -> list[str]:
    content = content.lstrip("\ufeff")
    modules: list[str] = []
    for pattern in (_FROM_IMPORT_PATTERN, _IMPORT_PATTERN):
        for match in pattern.finditer(content):
            module_name = match.group(1).strip()
            root_module = module_name.split(",")[0].strip()
            if root_module in modules:
                continue
            modules.append(root_module)
    return modules


def _looks_like_test_file(path: str) -> bool:
    filename = PurePosixPath(path).name
    return filename.startswith("test_") or "/tests/" in f"/{path}/"


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    return None
