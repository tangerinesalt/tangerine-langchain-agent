from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field


class LocalHTTPChatModel(SimpleChatModel):
    model_name: str
    url: str
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1)

    @property
    def _llm_type(self) -> str:
        return "local_http"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "url": self.url,
            "timeout_seconds": self.timeout_seconds,
        }

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [_message_to_payload(message) for message in messages],
        }
        if stop:
            payload["stop"] = stop
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]

        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_build_headers(self.api_key),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Local model request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ValueError(f"Local model request failed: {exc.reason}") from exc

        data = json.loads(body)
        return _extract_content(data)


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def _message_to_payload(message: BaseMessage) -> dict[str, str]:
    content = message.content if isinstance(message.content, str) else str(message.content)
    return {
        "role": message.type,
        "content": content,
    }


def _extract_content(data: dict[str, Any]) -> str:
    if isinstance(data.get("content"), str):
        return str(data["content"])
    if isinstance(data.get("text"), str):
        return str(data["text"])

    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])

    output = data.get("output")
    if isinstance(output, dict):
        if isinstance(output.get("content"), str):
            return str(output["content"])
        if isinstance(output.get("text"), str):
            return str(output["text"])

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict) and isinstance(choice_message.get("content"), str):
                return str(choice_message["content"])
            if isinstance(first_choice.get("text"), str):
                return str(first_choice["text"])

    raise ValueError("Local model response does not contain a supported text field.")
