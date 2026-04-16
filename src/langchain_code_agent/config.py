from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    model_backend: str = "langchain"
    model_provider: str | None = "openai"
    model: str = "qwen/qwen3.5-9b"
    model_api_key: str | None = "sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak"
    model_base_url: str | None = "http://localhost:1234/v1"
    model_timeout_seconds: int = 60
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
    ) -> AgentConfig:
        data: dict[str, Any] = {}
        if config_path is not None:
            with Path(config_path).open("rb") as handle:
                data = tomllib.load(handle)

        env_data = {
            "model_backend": os.getenv("LCA_MODEL_BACKEND"),
            "model_provider": os.getenv("LCA_MODEL_PROVIDER"),
            "model": os.getenv("LCA_MODEL"),
            "model_api_key": os.getenv("LCA_MODEL_API_KEY") or os.getenv("LCA_OPENAI_API_KEY"),
            "model_base_url": os.getenv("LCA_MODEL_BASE_URL"),
            "model_timeout_seconds": os.getenv("LCA_MODEL_TIMEOUT_SECONDS"),
            "log_level": os.getenv("LCA_LOG_LEVEL"),
        }
        data.update({key: value for key, value in env_data.items() if value})

        resolved_workspace = Path(
            workspace_root or data.get("workspace_root") or Path.cwd()
        ).expanduser().resolve()
        return cls(
            workspace_root=resolved_workspace,
            planner_backend=planner_backend or data.get("planner_backend", "noop"),
            model_backend=data.get("model_backend", "langchain"),
            model_provider=data.get("model_provider", "openai"),
            model=data.get("model", "qwen/qwen3.5-9b"),
            model_api_key=data.get("model_api_key", "sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak"),
            model_base_url=data.get("model_base_url", "http://localhost:1234/v1"),
            model_timeout_seconds=int(data.get("model_timeout_seconds", 60)),
            shell_timeout_seconds=int(data.get("shell_timeout_seconds", 30)),
            max_replans=int(data.get("max_replans", 1)),
            test_command=data.get("test_command"),
            log_level=data.get("log_level", "INFO"),
            ignore_patterns=list(data.get("ignore_patterns", DEFAULT_IGNORE_PATTERNS)),
            allowed_shell_commands=list(
                data.get("allowed_shell_commands", DEFAULT_ALLOWED_SHELL_COMMANDS)
            ),
        )
