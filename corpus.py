"""Two small knowledge bases + an eval set for Corrective RAG.

CRAG (Yan et al., 2024) distinguishes *internal* knowledge (your indexed corpus)
from *external* knowledge (the paper's large-scale web search, used as a fallback
when the internal retrieval is judged wrong). We model both as small, inspectable,
in-memory corpora so the whole corrective control flow runs offline and is unit
tested without a network — the external corpus stands in for "the web."

The eval questions are chosen to exercise all three corrective actions:
  * CORRECT   - well covered internally (answerable from INTERNAL_CORPUS).
  * INCORRECT - absent internally but present externally (forces a fallback).
  * AMBIGUOUS - thinly covered internally, completed by external knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    expect_action: str   # CORRECT | INCORRECT | AMBIGUOUS (the corrective path)
    note: str = ""


# The local, indexed knowledge base — solar-system facts, well covered.
INTERNAL_CORPUS: list[Document] = [
    Document("jupiter", "Jupiter",
             "Jupiter is the largest planet in the Solar System. It is a gas giant "
             "with a prominent storm called the Great Red Spot."),
    Document("mars", "Mars",
             "Mars is the fourth planet from the Sun. It has two small moons, Phobos "
             "and Deimos, and a reddish appearance from iron oxide."),
    Document("earth", "Earth",
             "Earth is the third planet from the Sun and the only known planet to "
             "support life. It has one natural satellite, the Moon."),
    Document("saturn", "Saturn",
             "Saturn is the sixth planet from the Sun and is famous for its extensive "
             "and bright ring system made largely of ice particles."),
    Document("sun", "The Sun",
             "The Sun is the star at the centre of the Solar System. It is composed "
             "mostly of hydrogen and helium."),
]

# The "external" source (a stand-in for web search) — broader, different topics.
EXTERNAL_CORPUS: list[Document] = [
    Document("shakespeare", "William Shakespeare",
             "William Shakespeare was an English playwright who wrote the tragedy "
             "Hamlet around 1600. He is widely regarded as the greatest writer in "
             "the English language."),
    Document("everest", "Mount Everest",
             "Mount Everest is Earth's highest mountain above sea level, with a peak "
             "at 8849 metres in the Himalayas."),
    Document("python", "Python (language)",
             "Python is a high-level programming language created by Guido van Rossum "
             "and first released in 1991."),
    # Overlaps Jupiter only thinly — used to complete an AMBIGUOUS internal hit.
    Document("great_red_spot", "Great Red Spot",
             "The Great Red Spot is a giant storm on Jupiter that has persisted for "
             "centuries and is large enough to swallow the Earth."),
]


EVAL_QUESTIONS: list[QAExample] = [
    QAExample("What is the largest planet in the Solar System?", "Jupiter",
              expect_action="CORRECT", note="well covered internally"),
    QAExample("How many moons does Mars have?", "two",
              expect_action="CORRECT"),
    QAExample("Who wrote the play Hamlet?", "Shakespeare",
              expect_action="INCORRECT", note="absent internally; falls back to external"),
    QAExample("What is the world's highest mountain?", "Everest",
              expect_action="INCORRECT", note="absent internally; external has it"),
    QAExample("How long has the storm on the largest planet lasted?", "centuries",
              expect_action="AMBIGUOUS", note="internal names the storm; external has its age"),
]


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def matches(predicted: str | None, example: QAExample) -> bool:
    if not predicted:
        return False
    return normalize(example.answer) in normalize(predicted)


__all__ = [
    "Document", "QAExample", "INTERNAL_CORPUS", "EXTERNAL_CORPUS",
    "EVAL_QUESTIONS", "matches", "normalize",
]
