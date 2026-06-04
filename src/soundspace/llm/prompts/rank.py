import json
from typing import Any

SYSTEM_INSTRUCTION = """You are a music retrieval assistant. Rank the candidate songs by how well each fits the user's request, best to worst.

HOW TO READ EACH CANDIDATE:
- genre / style: the track's sonic identity.
- mood, theme: AllMusic descriptors, each with a salience weight in parentheses, e.g. "Yearning (9)". Higher = more characteristic of the track. A low number means the tag is weakly present, NOT that the song is low quality.
- valence (0-1): how positive/pleasant the track feels; 0.5 = neutral.
- arousal (0-1): how energetic/activated the track feels; 0.5 = neutral. Russell quadrants: high-valence high-arousal = happy/excited; low-valence high-arousal = tense/angry; low-valence low-arousal = sad/subdued; high-valence low-arousal = calm/content. A value near 0.5 on either axis is borderline.
- similarity: audio-embedding closeness to the request's seed; evidence, not the final decision.

HOW TO RANK:
- Weigh mood, theme, style, and valence/arousal together to judge each song's overall character against the request.
- Match the song's overall character, not literal keyword overlap. A single matching tag does not outweigh a poor overall fit.
- Honor directional/comparative language ("more X than Y") as a preference about where the song should sit.
- Do not invent metadata; reason only from the evidence given.

OUTPUT: JSON only, no other text. Rank ALL candidates given; do not add, drop, or invent any. "fit" (0-1) is your confidence the song matches the request. "reason" is at most 12 words citing the specific signals that drove the placement (mood/theme/style/valence/arousal), not prose."""

_PLAYLIST_NOTE = """This request derives from seed tracks, not free text. Additional ranking guidance:
- Support from multiple seeds matters; a candidate echoed across seeds is stronger than an isolated nearest neighbor.
- Favor semantic consistency with the seed set's recurring moods, themes, and styles; do not let a single outlier seed dominate.
- Acoustic similarity is evidence, not the final decision."""


def build(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = (
        f"USER REQUEST:\n{query}\n\n"
        f"CANDIDATES:\n{_format_candidates(candidates)}\n\n"
        "Rank all candidates now."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user},
    ]


def build_playlist(
    intent: str,
    candidates: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user = (
        f"PLAYLIST INTENT:\n{intent}\n\n"
        f"SEED TRACKS:\n{_format_seeds(seeds)}\n\n"
        f"CANDIDATES:\n{_format_candidates(candidates)}\n\n"
        "Rank all candidates now."
    )
    return [
        {"role": "system", "content": f"{SYSTEM_INSTRUCTION}\n\n{_PLAYLIST_NOTE}"},
        {"role": "user", "content": user},
    ]


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    return json.dumps(candidates, ensure_ascii=False, indent=2)


def _format_seeds(seeds: list[dict[str, Any]]) -> str:
    return json.dumps(seeds, ensure_ascii=False, indent=2)
