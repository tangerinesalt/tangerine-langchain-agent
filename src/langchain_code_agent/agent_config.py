from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_code_agent.model_resolution import (
    DEFAULT_MODEL_TIMEOUT_SECONDS,
    load_workspace_config,
    resolve_model_settings,
)

DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
]

DEFAULT_ALLOWED_SHELL_COMMANDS = [
    "git",
    "python",
    "pytest",
    "rg",
]


@dataclass(slots=True)
class AgentConfig:
    workspace_root: Path
    planner_backend: str = "noop"
    model_profile: str | None = None
    model_backend: str = "langchain"
    model_provider: str | None = None
    model: str = ""
    model_api_key: str | None = None
    model_base_url: str | None = None
    model_timeout_seconds: int = DEFAULT_MODEL_TIMEOUT_SECONDS
    model_config_path: Path | None = None
    auth_path: Path | None = None
    model_api_key_source: str | None = None
    shell_timeout_seconds: int = 30
    max_replans: int = 1
    test_command: str | None = None
    log_level: str = "INFO"
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    allowed_shell_commands: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_SHELL_COMMANDS)
    )

    @classmethod
    def from_sources(
        cls,
        *,
        workspace_root: str | Path | None = None,
        config_path: str | Path | None = None,
        planner_backend: str | None = None,
        model_profile: str | None = None,
        global_model_config_path: str | Path | None = None,
        auth_path: str | Path | None = None,
    ) -> AgentConfig:
        workspace_data = load_workspace_config(config_path)
        resolved_planner_backend = planner_backend or workspace_data.get("planner_backend", "noop")
        model_settings = resolve_model_settings(
            profile_name=model_profile,
            model_config_path=global_model_config_path,
            auth_path=auth_path,
            require_model_settings=resolved_planner_backend == "langchain",
        )

        resolved_workspace = Path(
            workspace_root or workspace_data.get("workspace_root") or Path.cwd()
        ).expanduser().resolve()
        return cls(
            workspace_root=resolved_workspace,
            planner_backend=resolved_planner_backend,
            model_profile=model_settings.profile_name,
            model_backend=model_settings.model_backend or "langchain",
            model_provider=model_settings.model_provider,
            model=model_settings.model or "",
            model_api_key=model_settings.model_api_key,
            model_base_url=model_settings.model_base_url,
            model_timeout_seconds=model_settings.model_timeout_seconds,
            model_config_path=model_settings.model_config_path,
            auth_path=model_settings.auth_path,
            model_api_key_source=model_settings.model_api_key_source,
            shell_timeout_seconds=int(workspace_data.get("shell_timeout_seconds", 30)),
            max_replans=int(workspace_data.get("max_replans", 1)),
            test_command=workspace_data.get("test_command"),
            log_level=os.getenv("LCA_LOG_LEVEL") or workspace_data.get("log_level", "INFO"),
            ignore_patterns=list(workspace_data.get("ignore_patterns", DEFAULT_IGNORE_PATTERNS)),
            allowed_shell_commands=list(
                workspace_data.get("allowed_shell_commands", DEFAULT_ALLOWED_SHELL_COMMANDS)
            ),
        )
