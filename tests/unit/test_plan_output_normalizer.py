from pathlib import Path

from langchain_code_agent.agent.plan_normalization_rules import (
    apply_json_text_repairs,
    apply_plan_normalization_rules,
)
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.task import Task


def test_apply_json_text_repairs_strips_fences_extracts_json_and_escapes_backslashes() -> None:
    repaired = apply_json_text_repairs(
        "```json\n"
        'before {"summary":"x","steps":[{"action":"write_file","description":"d",'
        '"arguments":{"path":"a\\\\b.txt","content":"c"}}]} after\n'
        "```"
    )

    assert repaired.startswith('{"summary":"x"')
    assert repaired.endswith("]}")
    assert "a\\\\b.txt" in repaired


def test_apply_plan_normalization_rules_normalizes_dedupes_and_injects_time_anchor() -> None:
    plan = Plan(
        summary="  Move and write output.  ",
        steps=[
            PlanStep(
                action="find_files_by_name",
                description=" Find target. ",
                arguments={"query": "main"},
            ),
            PlanStep(
                action="find_files_by_name",
                description=" Find target. ",
                arguments={"query": "main"},
            ),
            PlanStep(
                action="move_file",
                description=" Move it. ",
                arguments={"src": "main.py", "dst": "pkg/main.py"},
            ),
        ],
    )
    task = Task(
        goal="move main.py today",
        workspace_root=Path.cwd(),
        execution_mode="execute",
    )

    normalized = apply_plan_normalization_rules(plan, task=task, workspace_root=Path.cwd())

    assert normalized.summary == "Move and write output."
    assert [step.action for step in normalized.steps] == [
        "get_current_date",
        "find_files_by_name",
        "move_file",
    ]
    assert normalized.steps[1].arguments == {"name": "main"}
    assert normalized.steps[2].arguments == {
        "source_path": "main.py",
        "destination_path": "pkg/main.py",
    }


def test_apply_plan_normalization_rules_collapses_empty_write_file_into_script_output() -> None:
    workspace_root = Path("C:/Users/tangerine/Desktop/Test/agentTest")
    plan = Plan(
        summary="Save output.",
        steps=[
            PlanStep(
                action="run_python_script",
                description="Generate text.",
                arguments={"script": "print('weather content')"},
            ),
            PlanStep(
                action="write_file",
                description="Write file.",
                arguments={
                    "path": "workspace\\agentTest\\weather_forecast.txt",
                    "content": "",
                    "overwrite": True,
                },
            ),
        ],
    )
    task = Task(
        goal="write the weather for the next three days into a txt file",
        workspace_root=workspace_root,
        execution_mode="execute",
    )

    normalized = apply_plan_normalization_rules(plan, task=task, workspace_root=workspace_root)

    assert [step.action for step in normalized.steps] == ["get_current_date", "run_python_script"]
    assert "write_text(_output_text" in str(normalized.steps[1].arguments["script"])
    assert "weather_forecast.txt" in str(normalized.steps[1].arguments["script"])


def test_apply_plan_normalization_rules_converts_run_command_command_to_run_shell() -> None:
    plan = Plan(
        summary="Run tests.",
        steps=[
            PlanStep(
                action="run_command",
                description="Run pytest.",
                arguments={"command": "python -m pytest -q"},
            )
        ],
    )
    task = Task(goal="fix failing tests", workspace_root=Path.cwd(), execution_mode="execute")

    normalized = apply_plan_normalization_rules(plan, task=task, workspace_root=Path.cwd())

    assert normalized.steps[0].action == "run_shell"
    assert normalized.steps[0].arguments == {"command": "python -m pytest -q"}


def test_apply_plan_normalization_rules_maps_file_alias_and_run_tests_path() -> None:
    workspace_root = Path("C:/Users/tangerine/Desktop/Test/agentTest/test")
    plan = Plan(
        summary="Read and test.",
        steps=[
            PlanStep(
                action="read_file",
                description="Read implementation.",
                arguments={"file": "math_utils.py"},
            ),
            PlanStep(
                action="run_tests",
                description="Run tests from workspace.",
                arguments={"path": "C:/Users/tangerine/Desktop/Test/agentTest/test"},
            ),
        ],
    )
    task = Task(goal="fix failing tests", workspace_root=workspace_root, execution_mode="execute")

    normalized = apply_plan_normalization_rules(plan, task=task, workspace_root=workspace_root)

    assert normalized.steps[0].arguments == {"path": "math_utils.py"}
    assert normalized.steps[1].arguments == {"working_directory": "."}
