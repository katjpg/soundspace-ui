from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderType(str, Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class LLMProviderError(Exception): ...


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage
    model: str
    provider: ProviderType
    finish_reason: str | None = None


class StructuredResponse(BaseModel, Generic[T]):
    parsed: T
    usage: TokenUsage
    model: str
    provider: ProviderType
    finish_reason: str | None = None


class LLMProvider(ABC):
    name: ProviderType

    @abstractmethod
    async def generate(self, *, messages: list[dict[str, str]]) -> LLMResponse: ...

    @abstractmethod
    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float | None = None,
    ) -> StructuredResponse[T]: ...
