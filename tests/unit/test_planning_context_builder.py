from pathlib import Path

from langchain_code_agent.actions import ActionRuntime
from langchain_code_agent.agent.planning.context_builder import build_planning_context
from langchain_code_agent.workspace.repository import Repository


def test_build_planning_context_extracts_fix_failing_tests_evidence() -> None:
    project_root = Path(__file__).resolve().parents[2]
    workspace = project_root / "tests" / "fixtures" / "agent_tasks" / "fixtures" / "failing_tests"
    repository = Repository(workspace, ignore_patterns=["__pycache__", ".pytest_cache"])
    runtime = ActionRuntime(
        repository=repository,
        workspace_root=workspace,
        shell_timeout_seconds=10,
        allowed_shell_commands=["python", "pytest"],
        test_command="python -m pytest -q",
    )

    planning_context = build_planning_context(
        "Fix the failing tests in this workspace.",
        execution_mode="execute",
        repository=repository,
        action_runtime=runtime,
    )

    assert planning_context is not None
    assert planning_context.fix_failing_tests is not None
    assert planning_context.fix_failing_tests.run_tests_ok is False
    assert planning_context.fix_failing_tests.run_tests_returncode == 1
    assert planning_context.fix_failing_tests.failing_test_ids == ["test_app.py::test_add"]
    assert planning_context.fix_failing_tests.candidate_paths == ["test_app.py", "app.py"]
    assert [item.path for item in planning_context.fix_failing_tests.file_excerpts] == [
        "test_app.py",
        "app.py",
    ]
    assert "return left - right" in planning_context.fix_failing_tests.file_excerpts[1].content
    assert any("app.py" in line for line in planning_context.fix_failing_tests.workspace_summary)
    assert any(
        "test_app.py" in line for line in planning_context.fix_failing_tests.workspace_summary
    )
    assert "FAILED test_app.py::test_add" in str(planning_context.fix_failing_tests.failure_excerpt)


def test_build_planning_context_skips_non_fix_failing_tests_task() -> None:
    project_root = Path(__file__).resolve().parents[2]
    workspace = project_root / "tests" / "fixtures" / "agent_tasks" / "fixtures" / "failing_tests"
    repository = Repository(workspace, ignore_patterns=["__pycache__", ".pytest_cache"])
    runtime = ActionRuntime(
        repository=repository,
        workspace_root=workspace,
        shell_timeout_seconds=10,
        allowed_shell_commands=["python", "pytest"],
        test_command="python -m pytest -q",
    )

    planning_context = build_planning_context(
        "List the repository files.",
        execution_mode="execute",
        repository=repository,
        action_runtime=runtime,
    )

    assert planning_context is None


def test_build_planning_context_extracts_imports_from_bom_test_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "\ufefffrom app import add\n\n\ndef test_add() -> None:\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    repository = Repository(tmp_path, ignore_patterns=["__pycache__", ".pytest_cache"])
    runtime = ActionRuntime(
        repository=repository,
        workspace_root=tmp_path,
        shell_timeout_seconds=10,
        allowed_shell_commands=["python", "pytest"],
        test_command="python -m pytest -q",
    )

    planning_context = build_planning_context(
        "Fix the failing tests in this workspace.",
        execution_mode="execute",
        repository=repository,
        action_runtime=runtime,
    )

    assert planning_context is not None
    assert planning_context.fix_failing_tests is not None
    assert planning_context.fix_failing_tests.candidate_paths == [
        "tests/test_app.py",
        "app.py",
    ]
