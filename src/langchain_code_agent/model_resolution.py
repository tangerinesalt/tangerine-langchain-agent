from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_TIMEOUT_SECONDS = 60

GLOBAL_AGENT_DIR_NAME = ".langchain-code-agent"
DEFAULT_GLOBAL_MODEL_CONFIG_BASENAME = "models.global.toml"
DEFAULT_GLOBAL_AUTH_BASENAME = "auth.json"

WORKSPACE_CONFIG_KEYS = {
    "workspace_root",
    "planner_backend",
    "shell_timeout_seconds",
    "max_replans",
    "test_command",
    "log_level",
    "ignore_patterns",
    "allowed_shell_commands",
}


@dataclass(slots=True)
class ResolvedModelSettings:
    profile_name: str | None
    model_backend: str | None
    model_provider: str | None
    model: str | None
    model_api_key: str | None
    model_base_url: str | None
    model_timeout_seconds: int
    model_config_path: Path
    auth_path: Path
    model_api_key_source: str | None


def default_global_model_config_path() -> Path:
    return Path.home() / GLOBAL_AGENT_DIR_NAME / DEFAULT_GLOBAL_MODEL_CONFIG_BASENAME


def default_auth_path() -> Path:
    return Path.home() / GLOBAL_AGENT_DIR_NAME / DEFAULT_GLOBAL_AUTH_BASENAME


def load_workspace_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    raw_data = _load_toml(Path(config_path))
    return {key: value for key, value in raw_data.items() if key in WORKSPACE_CONFIG_KEYS}


def resolve_model_settings(
    *,
    profile_name: str | None = None,
    model_config_path: str | Path | None = None,
    auth_path: str | Path | None = None,
    require_model_settings: bool = False,
) -> ResolvedModelSettings:
    resolved_model_config_path = Path(
        model_config_path
        or os.getenv("LCA_GLOBAL_MODEL_CONFIG")
        or default_global_model_config_path()
    ).expanduser()
    resolved_auth_path = Path(
        auth_path or os.getenv("LCA_AUTH_FILE") or default_auth_path()
    ).expanduser()

    model_config = _load_toml_if_exists(resolved_model_config_path)
    selected_profile = (
        profile_name or os.getenv("LCA_MODEL_PROFILE") or model_config.get("default_profile")
    )
    profile_values = _load_profile_values(model_config, selected_profile, resolved_model_config_path)

    env_api_key = os.getenv("LCA_MODEL_API_KEY") or os.getenv("LCA_OPENAI_API_KEY")
    api_key = env_api_key
    api_key_source = _api_key_env_source() if env_api_key else None
    if not api_key:
        api_key, api_key_source = _resolve_api_key_from_auth_file(
            auth_ref=_optional_str(profile_values.get("auth_ref")),
            auth_path=resolved_auth_path,
        )

    model_backend = _optional_str(os.getenv("LCA_MODEL_BACKEND")) or _optional_str(
        profile_values.get("model_backend")
    )
    model_provider = _optional_str(os.getenv("LCA_MODEL_PROVIDER")) or _optional_str(
        profile_values.get("model_provider")
    )
    model = _optional_str(os.getenv("LCA_MODEL")) or _optional_str(profile_values.get("model"))
    model_base_url = _optional_str(os.getenv("LCA_MODEL_BASE_URL")) or _optional_str(
        profile_values.get("model_base_url")
    )
    model_timeout_seconds = (
        _int_value(os.getenv("LCA_MODEL_TIMEOUT_SECONDS"))
        or _int_value(profile_values.get("model_timeout_seconds"))
        or DEFAULT_MODEL_TIMEOUT_SECONDS
    )

    if require_model_settings:
        if not model_backend:
            raise ValueError(
                "Model configuration is missing model_backend. Set it in models.global.toml, "
                "or provide LCA_MODEL_BACKEND."
            )
        if not model:
            raise ValueError(
                "Model configuration is missing model. Set it in models.global.toml, "
                "or provide LCA_MODEL."
            )

    return ResolvedModelSettings(
        profile_name=selected_profile,
        model_backend=model_backend,
        model_provider=model_provider,
        model=model,
        model_api_key=api_key,
        model_base_url=model_base_url,
        model_timeout_seconds=model_timeout_seconds,
        model_config_path=resolved_model_config_path.resolve(),
        auth_path=resolved_auth_path.resolve(),
        model_api_key_source=api_key_source,
    )


def _load_profile_values(
    model_config: dict[str, Any],
    profile_name: str | None,
    model_config_path: Path,
) -> dict[str, Any]:
    if not profile_name:
        return {}
    profiles = model_config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid profiles table in {model_config_path}.")
    profile_values = profiles.get(profile_name)
    if profile_values is None:
        raise ValueError(f"Model profile '{profile_name}' was not found in {model_config_path}.")
    if not isinstance(profile_values, dict):
        raise ValueError(
            f"Model profile '{profile_name}' in {model_config_path} must be a table."
        )
    return profile_values


def _resolve_api_key_from_auth_file(
    *,
    auth_ref: str | None,
    auth_path: Path,
) -> tuple[str | None, str | None]:
    if not auth_ref:
        return None, None
    auth_data = _load_json_if_exists(auth_path)
    credentials = auth_data.get("credentials", {})
    if not isinstance(credentials, dict):
        return None, f"missing:{auth_path}:credentials"
    credential = credentials.get(auth_ref)
    if not isinstance(credential, dict):
        return None, f"missing:{auth_path}:credentials.{auth_ref}"
    model_api_key = credential.get("model_api_key")
    if not isinstance(model_api_key, str) or not model_api_key:
        return None, f"missing:{auth_path}:credentials.{auth_ref}.model_api_key"
    return model_api_key, f"auth:{auth_path}:credentials.{auth_ref}.model_api_key"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a TOML table.")
    return data


def _load_toml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_toml(path)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Auth file {path} must contain a JSON object.")
    return data


def _api_key_env_source() -> str | None:
    if os.getenv("LCA_MODEL_API_KEY"):
        return "env:LCA_MODEL_API_KEY"
    if os.getenv("LCA_OPENAI_API_KEY"):
        return "env:LCA_OPENAI_API_KEY"
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return int(value)
    return None
