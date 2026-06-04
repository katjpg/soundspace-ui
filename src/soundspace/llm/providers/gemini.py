from dataclasses import dataclass
from typing import Any, TypeVar

from google import genai
from google.genai import types
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


@dataclass(frozen=True, slots=True)
class GeminiParams:
    api_key: str
    model: str
    max_output_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40
    thinking_level: str | None = None
    timeout: float = 60.0


def build_thinking_config(params: GeminiParams) -> types.ThinkingConfig | None:
    if not params.thinking_level:
        return None
    level = getattr(types.ThinkingLevel, params.thinking_level.upper())
    return types.ThinkingConfig(thinking_level=level)


def prepare_contents(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_instruction: str | None = None
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_instruction = text
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    return system_instruction, contents


def extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        total_tokens=getattr(usage, "total_token_count", 0) or 0,
    )


def extract_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return str(reason) if reason is not None else None


class GeminiProvider(LLMProvider):
    name = ProviderType.GEMINI

    def __init__(self, params: GeminiParams) -> None:
        if not params.api_key:
            raise ValueError("api_key is required")
        self._client = genai.Client(api_key=params.api_key)
        self._params = params

    def _build_config(
        self,
        *,
        temperature: float | None = None,
        **extra: Any,
    ) -> types.GenerateContentConfig:
        params = self._params
        kwargs: dict[str, Any] = dict(
            max_output_tokens=params.max_output_tokens,
            temperature=params.temperature if temperature is None else temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            **extra,
        )
        thinking_config = build_thinking_config(params)
        if thinking_config is not None:
            kwargs["thinking_config"] = thinking_config
        return types.GenerateContentConfig(**kwargs)

    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        **config_extra: Any,
    ) -> Any:
        system_instruction, contents = prepare_contents(messages)
        extra = dict(config_extra)
        if system_instruction is not None:
            extra["system_instruction"] = system_instruction
        gen_config = self._build_config(temperature=temperature, **extra)
        try:
            response = await self._client.aio.models.generate_content(
                model=self._params.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise LLMProviderError(f"Gemini call failed: {exc}") from exc
        if not response.text:
            raise LLMProviderError("Gemini returned an empty response")
        return response

    async def generate(self, *, messages: list[dict[str, str]]) -> LLMResponse:
        response = await self._call(messages)
        return LLMResponse(
            content=response.text,
            usage=extract_usage(response),
            model=self._params.model,
            provider=ProviderType.GEMINI,
            finish_reason=extract_finish_reason(response),
        )

    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float | None = None,
    ) -> StructuredResponse[T]:
        response = await self._call(
            messages,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        finish_reason = extract_finish_reason(response)
        if finish_reason and "MAX_TOKENS" in finish_reason:
            raise LLMProviderError(
                f"Gemini response truncated at "
                f"max_output_tokens={self._params.max_output_tokens}; raise the limit"
            )
        return StructuredResponse(
            parsed=schema.model_validate_json(response.text),
            usage=extract_usage(response),
            model=self._params.model,
            provider=ProviderType.GEMINI,
            finish_reason=finish_reason,
        )
