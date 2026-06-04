from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from soundspace.llm.providers.base import LLMProvider, StructuredResponse

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class AgentError(Exception): ...


class BaseAgent(BaseModel, Generic[ResponseT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    role: str
    provider: LLMProvider
    system_instruction: str
    max_retries: int = Field(default=3, ge=1)

    def build_input(self, context: dict[str, Any]) -> str:
        raise NotImplementedError("Subclasses must implement build_input.")

    def response_schema(self) -> type[ResponseT]:
        raise NotImplementedError("Subclasses must implement response_schema.")

    async def run(self, context: dict[str, Any]) -> ResponseT:
        raise NotImplementedError("Subclasses must implement run.")

    async def _generate(
        self, prompt: str, *, temperature: float | None = None
    ) -> StructuredResponse[ResponseT]:
        return await self.provider.generate_structured(
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": prompt},
            ],
            schema=self.response_schema(),
            temperature=temperature,
        )

    async def _with_retries(self, call: Callable[[], Awaitable[Any]]) -> Any:
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                return await call()
            except Exception as exc:
                last_error = exc
        raise AgentError(
            f"{self.name} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
