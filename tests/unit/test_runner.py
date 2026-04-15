from pathlib import Path

from langchain_code_agent.agent.runner import AgentRunner
from langchain_code_agent.config import AgentConfig
from langchain_code_agent.models.plan import Plan, PlanStep


def test_runner_dry_run_records_plan_without_execution() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"
    config = AgentConfig(
        workspace_root=sample_repo,
        planner_backend="noop",
        shell_timeout_seconds=10,
        test_command="python -m pytest -q",
        ignore_patterns=["__pycache__"],
        allowed_shell_commands=["python", "pytest"],
    )

    result = AgentRunner(config).run("check failing tests", execution_mode="dry-run")

    assert result.execution_mode == "dry-run"
    assert result.events
    assert result.events[0].event_type == "task_received"
    assert result.final_report.task_input["task"] == "check failing tests"
    assert result.final_report.plan_summary == "A minimal local execution plan."
    assert all(step.status == "planned" for step in result.step_results)
    assert all(step.ok is True for step in result.step_results)


def test_runner_execute_runs_steps() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"
    config = AgentConfig(
        workspace_root=sample_repo,
        planner_backend="noop",
        shell_timeout_seconds=10,
        test_command="python -m pytest -q",
        ignore_patterns=["__pycache__"],
        allowed_shell_commands=["python", "pytest"],
    )

    result = AgentRunner(config).run("check failing tests", execution_mode="execute")

    run_tests_result = next(step for step in result.step_results if step.action == "run_tests")
    assert result.execution_mode == "execute"
    assert run_tests_result.status == "completed"
    assert "passed" in str(run_tests_result.data["stdout"])
    assert result.final_report.shell_outputs
    assert result.final_report.tool_calls


def test_runner_records_error_context_for_failed_step() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"
    config = AgentConfig(
        workspace_root=sample_repo,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=["__pycache__"],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Force a failing test command.",
            steps=[PlanStep(action="run_tests", description="Run tests.", arguments={})],
        )
    )

    result = runner.run("force failure", execution_mode="execute")

    assert result.step_results[0].ok is False
    assert result.step_results[0].error_context is not None
    assert result.final_report.errors
    assert result.final_report.success is False


def test_runner_records_file_changes_from_shell_step(tmp_path: Path) -> None:
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Create a file via shell.",
            steps=[
                PlanStep(
                    action="run_shell",
                    description="Create a file.",
                    arguments={
                        "command": (
                            'python -c "from pathlib import Path; '
                            'Path(\'note.txt\').write_text(\'x\')"'
                        )
                    },
                )
            ],
        )
    )

    result = runner.run("create note", execution_mode="execute")

    assert result.step_results[0].file_changes
    assert result.step_results[0].file_changes[0].path == "note.txt"
    assert result.final_report.file_changes
    assert any(event.event_type == "file_changes_detected" for event in result.events)


def test_runner_execute_can_write_file_step(tmp_path: Path) -> None:
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Write a note.",
            steps=[
                PlanStep(
                    action="write_file",
                    description="Write the output file.",
                    arguments={"path": "weather.txt", "content": "sunny"},
                )
            ],
        )
    )

    result = runner.run("write weather file", execution_mode="execute")

    assert result.step_results[0].ok is True
    assert (tmp_path / "weather.txt").read_text(encoding="utf-8") == "sunny"
    assert result.final_report.file_changes
    assert result.final_report.file_changes[0].path == "weather.txt"


def test_runner_rejects_unknown_arguments_for_action(tmp_path: Path) -> None:
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Invalid list_files arguments.",
            steps=[
                PlanStep(
                    action="list_files",
                    description="Invalid arguments should fail.",
                    arguments={"path": "."},
                )
            ],
        )
    )

    result = runner.run("invalid args", execution_mode="execute")

    assert result.step_results[0].ok is False
    assert result.step_results[0].error_context is not None
    assert "does not accept arguments" in str(result.step_results[0].error)


def test_runner_execute_supports_phase1_repository_tools(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "main.py").write_text("value = 1\nvalue = 2\n", encoding="utf-8")
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Use repository discovery tools.",
            steps=[
                PlanStep(
                    action="glob_files",
                    description="Find Python files.",
                    arguments={"pattern": "**/*.py"},
                ),
                PlanStep(
                    action="find_files_by_name",
                    description="Find main files.",
                    arguments={"name": "main"},
                ),
                PlanStep(
                    action="tree_view",
                    description="Inspect directory tree.",
                    arguments={"path": ".", "depth": 2},
                ),
                PlanStep(
                    action="read_file_head",
                    description="Read the file header.",
                    arguments={"path": "pkg/main.py", "start_line": 1, "max_lines": 1},
                ),
                PlanStep(
                    action="search_text",
                    description="Search inside Python files.",
                    arguments={"query": "value", "path_glob": "pkg/*.py", "max_results": 2},
                ),
            ],
        )
    )

    result = runner.run("inspect repo deeply", execution_mode="execute")

    assert [step.action for step in result.step_results] == [
        "glob_files",
        "find_files_by_name",
        "tree_view",
        "read_file_head",
        "search_text",
    ]
    assert result.step_results[0].data["files"] == ["pkg/main.py"]
    assert result.step_results[3].data["content"] == "value = 1"
    assert result.step_results[4].data["count"] == 2


def test_runner_execute_supports_phase2_editing_tools(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("alpha\nbeta\n", encoding="utf-8")
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Edit files safely.",
            steps=[
                PlanStep(
                    action="insert_text",
                    description="Insert a new line.",
                    arguments={"path": "main.py", "anchor": "alpha\n", "text": "x\n"},
                ),
                PlanStep(
                    action="replace_in_file",
                    description="Replace existing text.",
                    arguments={"path": "main.py", "old_text": "beta", "new_text": "gamma"},
                ),
                PlanStep(
                    action="move_file",
                    description="Rename the file.",
                    arguments={"source_path": "main.py", "destination_path": "pkg/main.py"},
                ),
                PlanStep(
                    action="delete_file",
                    description="Delete the renamed file.",
                    arguments={"path": "pkg/main.py"},
                ),
            ],
        )
    )

    result = runner.run("edit file safely", execution_mode="execute")

    assert all(step.ok for step in result.step_results)
    assert any(change.change_type == "modified" for change in result.step_results[0].file_changes)
    assert any(change.change_type == "deleted" for change in result.step_results[-1].file_changes)
    assert not (tmp_path / "pkg" / "main.py").exists()


def test_runner_execute_supports_phase3_command_tools(tmp_path: Path) -> None:
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="noop",
        shell_timeout_seconds=5,
        ignore_patterns=[],
        allowed_shell_commands=["python"],
    )
    runner = AgentRunner(config)
    runner.planner = _StubPlanner(
        Plan(
            summary="Use stable execution tools.",
            steps=[
                PlanStep(
                    action="run_command",
                    description="Run argv form command.",
                    arguments={"argv": ["python", "-c", "print('argv')"]},
                ),
                PlanStep(
                    action="run_python_script",
                    description="Run multiline script.",
                    arguments={
                        "script": (
                            "from pathlib import Path\n"
                            "Path('from_script.txt').write_text('ok', encoding='utf-8')\n"
                            "print('script')\n"
                        )
                    },
                ),
            ],
        )
    )

    result = runner.run("run stable commands", execution_mode="execute")

    assert result.step_results[0].ok is True
    assert result.step_results[1].ok is True
    assert "argv" in str(result.step_results[0].data["stdout"])
    assert (tmp_path / "from_script.txt").read_text(encoding="utf-8") == "ok"
    assert result.final_report.shell_outputs


class _StubPlanner:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def create_plan(self, task) -> Plan:
        return self.plan
