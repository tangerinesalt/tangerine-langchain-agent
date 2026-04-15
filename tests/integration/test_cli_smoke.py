import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPO = PROJECT_ROOT / "tests" / "fixtures" / "sample_repo"


def test_cli_run_outputs_plan_and_results(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                f'workspace_root = "{SAMPLE_REPO.as_posix()}"',
                'planner_backend = "noop"',
                'shell_timeout_seconds = 30',
                'test_command = "python -m pytest -q"',
                'ignore_patterns = ["__pycache__"]',
                'allowed_shell_commands = ["python", "pytest", "rg", "git"]',
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "langchain_code_agent",
            "run",
            "--task",
            "check failing tests in sample repo",
            "--mode",
            "execute",
            "--config",
            str(config_path),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["planner"] == "noop"
    assert payload["execution_mode"] == "execute"
    assert payload["events"]
    assert payload["plan"]["steps"]
    assert payload["final_report"]["task_input"]["task"] == "check failing tests in sample repo"
    assert payload["final_report"]["shell_outputs"]
    assert any(item["action"] == "run_tests" for item in payload["step_results"])


def test_cli_dry_run_marks_steps_as_planned(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                f'workspace_root = "{SAMPLE_REPO.as_posix()}"',
                'planner_backend = "noop"',
                'shell_timeout_seconds = 30',
                'test_command = "python -m pytest -q"',
                'ignore_patterns = ["__pycache__"]',
                'allowed_shell_commands = ["python", "pytest"]',
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "langchain_code_agent",
            "run",
            "--task",
            "check failing tests in sample repo",
            "--mode",
            "dry-run",
            "--config",
            str(config_path),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["execution_mode"] == "dry-run"
    assert payload["final_report"]["planned_steps"] == len(payload["step_results"])
    assert all(item["status"] == "planned" for item in payload["step_results"])
