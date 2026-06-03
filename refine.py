"""Knowledge refinement — CRAG's decompose-then-recompose step.

A retrieved document, even a relevant one, usually mixes a few useful facts with
filler. CRAG decomposes each document into fine-grained *knowledge strips*
(sentences here), grades every strip for relevance, drops the irrelevant ones, and
recomposes the survivors. The generator then sees concentrated knowledge instead of
whole noisy passages.

Grading reuses the retrieval evaluator; splitting and recomposing are pure.
"""

from __future__ import annotations

import re

from .evaluator import grade
from .llm import chat

_SENT = re.compile(r"(?<=[.!?])\s+")


def into_strips(text: str) -> list[str]:
    """Split a passage into knowledge strips (sentences)."""
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def refine_doc(
    question: str,
    text: str,
    chat_fn=chat,
    model: str | None = None,
    keep: int = 1,
) -> str:
    """Keep only the strips that score at least ``keep`` for the question.

    If every strip is filtered out, fall back to the single best-scoring strip so a
    document judged relevant overall never refines down to nothing.
    """
    strips = into_strips(text)
    if not strips:
        return ""

    scored = [(s, grade(question, s, chat_fn=chat_fn, model=model)) for s in strips]
    kept = [s for s, sc in scored if sc >= keep]
    if not kept:
        kept = [max(scored, key=lambda x: x[1])[0]]
    return " ".join(kept)


__all__ = ["into_strips", "refine_doc"]
