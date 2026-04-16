from pathlib import Path

from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.llm import factory
from langchain_code_agent.llm.local_http import LocalHTTPChatModel


def test_build_chat_model_uses_provider_and_model(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None, dict[str, object]]] = []

    def fake_init_chat_model(
        model: str | None = None,
        *,
        model_provider: str | None = None,
        **kwargs: object,
    ) -> str:
        calls.append((model, model_provider, kwargs))
        return "fake-model"

    monkeypatch.setattr(factory, "init_chat_model", fake_init_chat_model)
    config = AgentConfig(
        workspace_root=Path.cwd(),
        model_backend="langchain",
        model_provider="openai",
        model="gpt-4o-mini",
        model_api_key="test-key",
    )

    model = factory.build_chat_model(config)

    assert model == "fake-model"
    assert calls[0][0] == "gpt-4o-mini"
    assert calls[0][1] == "openai"


def test_build_chat_model_accepts_prefixed_model(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None, dict[str, object]]] = []

    def fake_init_chat_model(
        model: str | None = None,
        *,
        model_provider: str | None = None,
        **kwargs: object,
    ) -> str:
        calls.append((model, model_provider, kwargs))
        return "prefixed-model"

    monkeypatch.setattr(factory, "init_chat_model", fake_init_chat_model)
    config = AgentConfig(
        workspace_root=Path.cwd(),
        model_backend="langchain",
        model_provider="openai",
        model="openai:gpt-4o-mini",
        model_api_key="test-key",
    )

    model = factory.build_chat_model(config)

    assert model == "prefixed-model"
    assert calls[0][0] == "openai:gpt-4o-mini"
    assert calls[0][1] is None


def test_build_chat_model_supports_local_http_backend() -> None:
    config = AgentConfig(
        workspace_root=Path.cwd(),
        model_backend="local_http",
        model_provider=None,
        model="qwen/qwen3.5-9b",
        model_base_url="http://localhost:1234/api/v1/chat",
        model_api_key="sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak",
        model_timeout_seconds=45,
    )

    model = factory.build_chat_model(config)

    assert isinstance(model, LocalHTTPChatModel)
    assert model.url == "http://localhost:1234/api/v1/chat"
    assert model.api_key == "sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak"
    assert model.timeout_seconds == 45
