from pathlib import Path

from langchain_core.messages import AIMessage

from langchain_code_agent.agent import planner as planner_module
from langchain_code_agent.config import AgentConfig
from langchain_code_agent.models.plan import Plan, PlanStep
from langchain_code_agent.models.task import Task


class _FakeAgent:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def invoke(self, _: dict[str, object]) -> dict[str, object]:
        return self.response


def test_langchain_planner_uses_structured_response(monkeypatch) -> None:
    expected_plan = Plan(
        summary="Inspect then test.",
        steps=[
            PlanStep(
                action="list_files",
                description="Inspect files.",
                arguments={"limit": 20},
            )
        ],
    )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: "fake-model")
    monkeypatch.setattr(
        planner_module,
        "create_agent",
        lambda **kwargs: _FakeAgent({"structured_response": expected_plan}),
    )

    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="https://api.openai.com/v1",
    )
    task = Task(goal="inspect repo", workspace_root=Path.cwd(), execution_mode="dry-run")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan == expected_plan


def test_langchain_planner_validates_mapping_response(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: "fake-model")
    monkeypatch.setattr(
        planner_module,
        "create_agent",
        lambda **kwargs: _FakeAgent(
            {
                "structured_response": {
                    "summary": "Run tests after inspection.",
                    "steps": [
                        {
                            "action": "run_tests",
                            "description": "Run tests.",
                            "arguments": {},
                        }
                    ],
                }
            }
        ),
    )

    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="https://api.openai.com/v1",
    )
    task = Task(goal="run tests", workspace_root=Path.cwd(), execution_mode="execute")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.summary == "Run tests after inspection."
    assert plan.steps[0].action == "run_tests"


def test_langchain_planner_falls_back_for_local_http_backend(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Inspect local repo.","steps":['
                    '{"action":"list_files","description":"Inspect files.",'
                    '"arguments":{"limit":50}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="local_http",
        model_provider=None,
        model="qwen/qwen3.5-9b",
        model_api_key="sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak",
        model_base_url="http://localhost:1234/api/v1/chat",
    )
    task = Task(goal="inspect repo", workspace_root=Path.cwd(), execution_mode="dry-run")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.summary == "Inspect local repo."
    assert plan.steps[0].action == "list_files"


def test_langchain_planner_falls_back_for_local_openai_compatible_endpoint(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Use deterministic local planning.","steps":['
                    '{"action":"list_files","description":"Inspect files first.",'
                    '"arguments":{"limit":25}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())

    def fail_create_agent(**kwargs):
        raise AssertionError("create_agent should not be used for local OpenAI-compatible URLs")

    monkeypatch.setattr(planner_module, "create_agent", fail_create_agent)
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model="qwen/qwen3.5-9b",
        model_api_key="sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(goal="inspect repo", workspace_root=Path.cwd(), execution_mode="dry-run")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.summary == "Use deterministic local planning."
    assert plan.steps[0].action == "list_files"


def test_langchain_planner_extracts_fenced_json_and_write_file(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    "```json\n"
                    '{\n'
                    '  "summary": "Write a text file.",\n'
                    '  "steps": [\n'
                    '    {\n'
                    '      "action": "write_file",\n'
                    '      "description": "Create the output file.",\n'
                    '      "arguments": {"path": "weather.txt", "content": "sunny"}\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```"
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(goal="write weather", workspace_root=Path.cwd(), execution_mode="execute")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.summary == "Write a text file."
    assert [step.action for step in plan.steps] == ["write_file"]
    assert plan.steps[0].arguments["path"] == "weather.txt"


def test_langchain_planner_adds_date_anchor_for_time_sensitive_task(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Write the weather file.","steps":['
                    '{"action":"write_file","description":"Write the output file.",'
                    '"arguments":{"path":"weather.txt","content":"sunny"}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(
        goal="write the weather for the next three days into a txt file",
        workspace_root=Path.cwd(),
        execution_mode="execute",
    )

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.steps[0].action == "get_current_date"
    assert plan.steps[0].arguments == {}
    assert plan.steps[1].action == "write_file"


def test_langchain_planner_normalizes_python_c_shell_to_run_python_script(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Run a simple script.","steps":['
                    '{"action":"run_shell","description":"Use python -c.",'
                    '"arguments":{"command":"python -c print(1)"}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(goal="print a value", workspace_root=Path.cwd(), execution_mode="execute")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert plan.steps[0].action == "run_python_script"
    assert "print(1)" in str(plan.steps[0].arguments["script"])


def test_langchain_planner_deduplicates_and_normalizes_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Inspect and move the file.","steps":['
                    '{"action":"find_files_by_name","description":"Find target.",'
                    '"arguments":{"query":"main"}},'
                    '{"action":"find_files_by_name","description":"Find target.",'
                    '"arguments":{"query":"main"}},'
                    '{"action":"move_file","description":"Move it.",'
                    '"arguments":{"src":"main.py","dst":"pkg/main.py"}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    config = AgentConfig(
        workspace_root=tmp_path,
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(goal="move main.py", workspace_root=tmp_path, execution_mode="execute")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert len(plan.steps) == 2
    assert plan.steps[0].action == "find_files_by_name"
    assert plan.steps[0].arguments == {"name": "main"}
    assert plan.steps[1].action == "move_file"
    assert plan.steps[1].arguments == {
        "source_path": "main.py",
        "destination_path": "pkg/main.py",
    }


def test_langchain_planner_retries_invalid_json_once(monkeypatch) -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list[object]) -> AIMessage:
            self.calls += 1
            assert messages
            if self.calls == 1:
                return AIMessage(
                    content=(
                        '{"summary":"Write the file.","steps":['
                        '{"action":"write_file","description":"Write it.",'
                        '"arguments":{"path":"notes.txt","content":"hello"}}'
                    )
                )
            return AIMessage(
                content=(
                    '{"summary":"Write the file.","steps":['
                    '{"action":"write_file","description":"Write it.",'
                    '"arguments":{"path":"notes.txt","content":"hello"}}'
                    "]}",
                )
            )

    fake_model = _FakeModel()
    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: fake_model)
    config = AgentConfig(
        workspace_root=Path.cwd(),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(goal="write notes.txt", workspace_root=Path.cwd(), execution_mode="execute")

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert fake_model.calls == 2
    assert plan.steps[0].action == "write_file"
    assert plan.steps[0].arguments["path"] == "notes.txt"


def test_langchain_planner_collapses_empty_write_file_into_script_output(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages
            return AIMessage(
                content=(
                    '{"summary":"Fetch weather and save it.","steps":['
                    '{"action":"run_python_script","description":"Fetch and print weather.",'
                    '"arguments":{"script":"print(\'weather content\')"}},'
                    '{"action":"write_file","description":"Write weather file.",'
                    '"arguments":{"path":"workspace\\\\agentTest\\\\weather_forecast.txt",'
                    '"content":"","overwrite":true}}'
                    "]}",
                )
            )

    monkeypatch.setattr(planner_module, "build_chat_model", lambda config: _FakeModel())
    config = AgentConfig(
        workspace_root=Path("C:/Users/tangerine/Desktop/Test/agentTest"),
        planner_backend="langchain",
        model_backend="langchain",
        model_provider="openai",
        model_base_url="http://localhost:1234/v1",
    )
    task = Task(
        goal="write the weather for the next three days into a txt file",
        workspace_root=Path("C:/Users/tangerine/Desktop/Test/agentTest"),
        execution_mode="execute",
    )

    plan = planner_module.LangChainPlanner(config).create_plan(task)

    assert [step.action for step in plan.steps] == ["get_current_date", "run_python_script"]
    assert "write_text(_output_text" in str(plan.steps[1].arguments["script"])
    assert "weather_forecast.txt" in str(plan.steps[1].arguments["script"])


def test_planner_system_prompt_guides_identity_time_and_conciseness() -> None:
    prompt = planner_module.PLANNER_SYSTEM_PROMPT

    assert "not a general chat assistant" in prompt
    assert "exact current date" in prompt
    assert "short summaries and short step descriptions" in prompt
    assert "absolute dates" in prompt
    assert "replace_in_file" in prompt
    assert "move_file" in prompt
    assert "get_current_date" in prompt
    assert "run_command" in prompt
    assert "run_python_script" in prompt
    assert "do not return only repository inspection steps" in prompt


def test_build_task_request_content_includes_replan_context_json() -> None:
    from langchain_code_agent.models.replan import ReplanContext, ReplanFailedStep

    task = Task(
        goal="write notes.txt",
        workspace_root=Path.cwd(),
        execution_mode="execute",
        replan_context=ReplanContext(
            original_task="write notes.txt",
            attempt=1,
            previous_plan_summary="Inspect only.",
            failed_steps=[
                ReplanFailedStep(
                    action="read_file_head",
                    message="Path does not exist: pytest.log",
                    arguments={"path": "pytest.log"},
                )
            ],
            completion_failures=["Expected file to exist after run: notes.txt"],
            successful_actions=["list_files"],
            file_changes=[],
        ),
    )

    content = planner_module._build_task_request_content(task)

    assert "Replan context JSON" in content
    assert '"action": "read_file_head"' in content
    assert '"completion_failures"' in content
