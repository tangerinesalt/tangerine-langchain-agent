from io import BytesIO

from langchain_core.messages import HumanMessage

from langchain_code_agent.llm.local_http import LocalHTTPChatModel


class _FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self.payload = BytesIO(payload.encode("utf-8"))

    def read(self) -> bytes:
        return self.payload.read()

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_local_http_model_reads_choice_message(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        assert request.full_url == "http://localhost:1234/api/v1/chat"
        assert timeout == 30
        return _FakeHTTPResponse('{"choices":[{"message":{"content":"hello from local"}}]}')

    monkeypatch.setattr("langchain_code_agent.llm.local_http.urlopen", fake_urlopen)
    model = LocalHTTPChatModel(
        model_name="qwen/qwen3.5-9b",
        url="http://localhost:1234/api/v1/chat",
        api_key="sk-lm-cgCA7T00:aaSvq1VWIISDxan6Niak",
        timeout_seconds=30,
    )

    response = model.invoke([HumanMessage(content="say hello")])

    assert response.content == "hello from local"


def test_local_http_model_reads_direct_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "langchain_code_agent.llm.local_http.urlopen",
        lambda request, timeout: _FakeHTTPResponse('{"content":"plain content"}'),
    )
    model = LocalHTTPChatModel(
        model_name="qwen/qwen3.5-9b",
        url="http://localhost:1234/api/v1/chat",
    )

    response = model.invoke([HumanMessage(content="say hello")])

    assert response.content == "plain content"
