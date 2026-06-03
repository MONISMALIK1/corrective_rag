"""The retrieval evaluator — CRAG's decision core.

It scores each retrieved internal document for relevance (0-3) and, from the best
score, picks one of three corrective actions:

  * CORRECT   - at least one document is clearly relevant (score >= UPPER).
                Trust internal knowledge (after refinement).
  * INCORRECT - no document is relevant at all (best score < LOWER).
                Discard internal knowledge; fall back to the external source.
  * AMBIGUOUS - in between. Combine refined internal knowledge with external.

Scoring is an LLM call (injectable ``chat_fn``); the parsing and the thresholding
are pure functions, so the decision logic is fully unit-tested offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import chat
from .prompts import RELEVANCE_SCORE_PROMPT

CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
AMBIGUOUS = "AMBIGUOUS"
ACTIONS = (CORRECT, INCORRECT, AMBIGUOUS)

# Score thresholds on the 0-3 relevance scale.
UPPER = 2   # best score >= UPPER  -> confident the internal docs are relevant
LOWER = 1   # best score <  LOWER  -> confident they are all irrelevant

_SCORE_RE = re.compile(r"score\s*[:=]?\s*(-?\d+)", re.IGNORECASE)


def parse_score(text: str) -> int:
    """Pull a 0-3 relevance score out of a reply; default 0 (fail toward 'irrelevant').

    Reads the last line first (where the prompt asks for it), then anywhere.
    """
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        m = _SCORE_RE.search(line)
        if m:
            return max(0, min(3, int(m.group(1))))
    m = _SCORE_RE.search(text)
    if m:
        return max(0, min(3, int(m.group(1))))
    return 0


def grade(question: str, passage: str, chat_fn=chat, model: str | None = None) -> int:
    """Relevance score (0-3) of one passage for the question, via the model."""
    prompt = RELEVANCE_SCORE_PROMPT.format(question=question, passage=passage)
    return parse_score(chat_fn(prompt, model=model))


def decide_action(scores: list[int]) -> str:
    """Map per-document relevance scores to a corrective action."""
    best = max(scores) if scores else 0
    if best >= UPPER:
        return CORRECT
    if best < LOWER:
        return INCORRECT
    return AMBIGUOUS


@dataclass
class GradedDoc:
    """A retrieved document paired with its relevance score."""
    doc_id: str
    title: str
    text: str
    score: int


__all__ = [
    "CORRECT", "INCORRECT", "AMBIGUOUS", "ACTIONS", "UPPER", "LOWER",
    "parse_score", "grade", "decide_action", "GradedDoc",
]
