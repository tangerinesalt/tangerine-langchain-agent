from pathlib import Path

from langchain_code_agent.config import AgentConfig


def test_config_loads_from_toml(tmp_path: Path) -> None:
    defaults = AgentConfig(workspace_root=tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                f'workspace_root = "{tmp_path.as_posix()}"',
                'planner_backend = "noop"',
                f'model_backend = "{defaults.model_backend}"',
                f'model_provider = "{defaults.model_provider}"',
                f'model = "{defaults.model}"',
                f'model_api_key = "{defaults.model_api_key}"',
                f'model_base_url = "{defaults.model_base_url}"',
                'test_command = "python -m pytest -q"',
            ]
        ),
        encoding="utf-8",
    )

    config = AgentConfig.from_sources(config_path=config_path)

    assert config.workspace_root == tmp_path.resolve()
    assert config.model_backend == defaults.model_backend
    assert config.model_provider == defaults.model_provider
    assert config.model == defaults.model
    assert config.model_api_key == defaults.model_api_key
    assert config.model_base_url == defaults.model_base_url
    assert config.test_command == "python -m pytest -q"
    assert "python" in config.allowed_shell_commands
