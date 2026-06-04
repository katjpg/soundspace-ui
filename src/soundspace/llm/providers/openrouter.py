from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from soundspace.llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderType,
    StructuredResponse,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)

_COMPLETIONS_PATH = "/chat/completions"


@dataclass(frozen=True, slots=True)
class OpenRouterParams:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 0.95
    reasoning_enabled: bool = True
    referer: str | None = None
    title: str | None = None
    timeout: float = 60.0


def build_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": schema.model_json_schema(),
        },
    }


def extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMProviderError("OpenRouter response had no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise LLMProviderError("OpenRouter returned an empty response")
    return content


def extract_usage(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage") or {}
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
    )


def extract_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    reason = choices[0].get("finish_reason")
    return str(reason) if reason is not None else None


class OpenRouterProvider(LLMProvider):
    name = ProviderType.OPENROUTER

    def __init__(
        self, params: OpenRouterParams, *, client: httpx.AsyncClient | None = None
    ) -> None:
        if not params.api_key:
            raise ValueError("api_key is required")
        self._params = params
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OpenRouterProvider:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._params.base_url, timeout=self._params.timeout
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._params.api_key}",
            "Content-Type": "application/json",
        }
        if self._params.referer:
            headers["HTTP-Referer"] = self._params.referer
        if self._params.title:
            headers["X-Title"] = self._params.title
        return headers

    def _body(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._params.model,
            "messages": messages,
            "max_tokens": self._params.max_tokens,
            "temperature": self._params.temperature
            if temperature is None
            else temperature,
            "top_p": self._params.top_p,
        }
        if self._params.reasoning_enabled:
            body["reasoning"] = {"enabled": True}
        if response_format is not None:
            body["response_format"] = response_format
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise LLMProviderError(
                "provider not started; use 'async with OpenRouterProvider()'"
            )
        try:
            response = await self._client.post(
                _COMPLETIONS_PATH, json=body, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"OpenRouter call failed: {exc}") from exc
        if response.status_code >= 400:
            raise LLMProviderError(
                f"OpenRouter http {response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMProviderError("expected OpenRouter response to be a JSON object")
        if "error" in payload:
            raise LLMProviderError(f"OpenRouter error: {payload['error']}")
        return payload

    async def generate(self, *, messages: list[dict[str, str]]) -> LLMResponse:
        payload = await self._post(
            self._body(messages, temperature=None, response_format=None)
        )
        return LLMResponse(
            content=extract_text(payload),
            usage=extract_usage(payload),
            model=self._params.model,
            provider=ProviderType.OPENROUTER,
            finish_reason=extract_finish_reason(payload),
        )

    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float | None = None,
    ) -> StructuredResponse[T]:
        body = self._body(
            messages,
            temperature=temperature,
            response_format=build_response_format(schema),
        )
        payload = await self._post(body)
        finish_reason = extract_finish_reason(payload)
        if finish_reason == "length":
            raise LLMProviderError(
                f"OpenRouter response truncated at max_tokens={self._params.max_tokens}; raise the limit"
            )
        return StructuredResponse(
            parsed=schema.model_validate_json(extract_text(payload)),
            usage=extract_usage(payload),
            model=self._params.model,
            provider=ProviderType.OPENROUTER,
            finish_reason=finish_reason,
        )
