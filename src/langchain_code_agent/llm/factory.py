from __future__ import annotations

from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.llm.local_http import LocalHTTPChatModel


def build_chat_model(config: AgentConfig) -> BaseChatModel:
    if config.model_backend == "local_http":
        if not config.model_base_url:
            raise ValueError("Local HTTP backend requires model_base_url.")
        return LocalHTTPChatModel(
            model_name=config.model,
            url=config.model_base_url,
            api_key=config.model_api_key,
            timeout_seconds=config.model_timeout_seconds,
        )

    kwargs: dict[str, Any] = {}
    if config.model_api_key:
        kwargs["api_key"] = SecretStr(config.model_api_key)
    if config.model_base_url:
        kwargs["base_url"] = config.model_base_url
        
    if config.model_provider:
        return cast(
            BaseChatModel,
            init_chat_model(config.model, model_provider=config.model_provider, **kwargs),
        )
    if ":" in config.model:
        return cast(BaseChatModel, init_chat_model(config.model, **kwargs))
    
    return cast(BaseChatModel, init_chat_model(config.model, **kwargs))
