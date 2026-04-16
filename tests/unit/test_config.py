import json
from pathlib import Path

import pytest

from langchain_code_agent.agent_config import AgentConfig


def test_config_loads_workspace_and_global_model_config_separately(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    config_path = tmp_path / "agent.config.toml"
    config_path.write_text(
        "\n".join(
            [
                f'workspace_root = "{workspace_root.as_posix()}"',
                'planner_backend = "noop"',
                'model = "should-be-ignored"',
                'model_api_key = "should-be-ignored"',
                'test_command = "python -m pytest -q"',
            ]
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "models.global.toml"
    model_config_path.write_text(
        "\n".join(
            [
                'default_profile = "local_qwen"',
                "",
                "[profiles.local_qwen]",
                'model_backend = "langchain"',
                'model_provider = "openai"',
                'model = "qwen/qwen3.5-9b"',
                'model_base_url = "http://localhost:1234/v1"',
                "model_timeout_seconds = 60",
                'auth_ref = "lmstudio_local"',
            ]
        ),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "credentials": {
                    "lmstudio_local": {
                        "model_api_key": "test-key",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = AgentConfig.from_sources(
        config_path=config_path,
        global_model_config_path=model_config_path,
        auth_path=auth_path,
    )

    assert config.workspace_root == workspace_root.resolve()
    assert config.model_profile == "local_qwen"
    assert config.model == "qwen/qwen3.5-9b"
    assert config.model_api_key == "test-key"
    assert config.model_api_key_source == (
        f"auth:{auth_path.resolve()}:credentials.lmstudio_local.model_api_key"
    )
    assert config.test_command == "python -m pytest -q"
    assert "python" in config.allowed_shell_commands


def test_config_uses_env_api_key_override(tmp_path: Path, monkeypatch) -> None:
    model_config_path = tmp_path / "models.global.toml"
    model_config_path.write_text(
        "\n".join(
            [
                'default_profile = "local_qwen"',
                "",
                "[profiles.local_qwen]",
                'model = "qwen/qwen3.5-9b"',
                'auth_ref = "lmstudio_local"',
            ]
        ),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "credentials": {
                    "lmstudio_local": {
                        "model_api_key": "file-key",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LCA_MODEL_API_KEY", "env-key")

    config = AgentConfig.from_sources(
        workspace_root=tmp_path,
        global_model_config_path=model_config_path,
        auth_path=auth_path,
    )

    assert config.model_api_key == "env-key"
    assert config.model_api_key_source == "env:LCA_MODEL_API_KEY"


def test_config_requires_model_data_for_langchain_planner(tmp_path: Path) -> None:
    model_config_path = tmp_path / "models.global.toml"
    model_config_path.write_text(
        "\n".join(
            [
                'default_profile = "local_qwen"',
                "",
                "[profiles.local_qwen]",
                'model_timeout_seconds = 60',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing model_backend"):
        AgentConfig.from_sources(
            workspace_root=tmp_path,
            planner_backend="langchain",
            global_model_config_path=model_config_path,
        )
