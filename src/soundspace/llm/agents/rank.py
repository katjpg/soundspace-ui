from typing import Any

from pydantic import BaseModel, Field

from soundspace.llm.agents.base import AgentError, BaseAgent
from soundspace.llm.prompts import rank as rank_prompt
from soundspace.llm.providers.base import LLMProviderError

_REPAIR_NOTE = (
    "Your previous response was not valid against the required schema: {error}. "
    "Return ONLY a JSON object matching the schema, ranking every candidate."
)


class RankedSong(BaseModel):
    rank: int
    id: str
    fit: float
    reason: str


class Ranking(BaseModel):
    items: list[RankedSong] = Field(default_factory=list)


class RankAgent(BaseAgent[Ranking]):
    name: str = "rank"
    role: str = "Ranks candidate tracks against a request by overall fit."
    system_instruction: str = rank_prompt.SYSTEM_INSTRUCTION
    max_retries: int = 2

    def response_schema(self) -> type[Ranking]:
        return Ranking

    async def rank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> list[RankedSong]:
        messages = rank_prompt.build(query, candidates)
        return await self._rank(messages, candidates, temperature=temperature)

    async def rank_playlist(
        self,
        intent: str,
        candidates: list[dict[str, Any]],
        seeds: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> list[RankedSong]:
        messages = rank_prompt.build_playlist(intent, candidates, seeds)
        return await self._rank(messages, candidates, temperature=temperature)

    async def _rank(
        self,
        messages: list[dict[str, str]],
        candidates: list[dict[str, Any]],
        *,
        temperature: float | None,
    ) -> list[RankedSong]:
        valid_ids = {str(c["id"]) for c in candidates}
        try:
            ranking = await self._call(messages, temperature=temperature)
            return _check_ranking(ranking, valid_ids)
        except (LLMProviderError, AgentError, ValueError) as first:
            repair = [
                *messages,
                {"role": "user", "content": _REPAIR_NOTE.format(error=first)},
            ]
            try:
                ranking = await self._call(repair, temperature=temperature)
                return _check_ranking(ranking, valid_ids)
            except (LLMProviderError, AgentError, ValueError) as second:
                raise AgentError(f"rank failed after repair: {second}") from second

    async def _call(
        self, messages: list[dict[str, str]], *, temperature: float | None
    ) -> Ranking:
        response = await self.provider.generate_structured(
            messages=messages, schema=Ranking, temperature=temperature
        )
        return response.parsed


def _check_ranking(ranking: Ranking, valid_ids: set[str]) -> list[RankedSong]:
    if not ranking.items:
        raise ValueError("ranking returned no items")
    unknown = [r.id for r in ranking.items if r.id not in valid_ids]
    if unknown:
        raise ValueError(f"ranking returned unknown ids: {unknown}")
    return sorted(ranking.items, key=lambda r: r.rank)
