"""The Corrective RAG control flow.

Reference: Yan, Gu, Zhu, Ling, Cai, Zhang 2024, "Corrective Retrieval-Augmented
Generation", https://arxiv.org/abs/2401.15884

Where Self-RAG reflects on retrieved passages and *abstains* when they look weak,
CRAG goes a step further and *corrects* the retrieval::

    retrieve from the internal index (BM25)
        |
    grade every document  (the retrieval evaluator, 0-3)
        |
    decide the corrective action from the best score:
        CORRECT   -> refine the relevant internal docs (decompose-recompose)
        INCORRECT -> discard them; fall back to the external knowledge source
        AMBIGUOUS -> combine refined internal knowledge with external
        |
    assemble the (refined) knowledge, with citations
        |
    generate a grounded, cited answer  -- or abstain if nothing usable remains

Only the grading, refinement, and generation steps touch the network (via the
injectable ``chat_fn``); retrieval and the action decision are pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evaluator import (AMBIGUOUS, CORRECT, INCORRECT, GradedDoc, decide_action,
                        grade)
from .llm import chat
from .prompts import GENERATE_PROMPT
from .refine import refine_doc
from .retriever import BM25Retriever


@dataclass
class Source:
    """One refined knowledge block handed to the generator, with its origin."""
    origin: str          # "internal" or "external"
    doc_id: str
    title: str
    text: str            # refined (recomposed) knowledge


@dataclass
class CRAGResult:
    query: str
    answer: str | None
    abstained: bool
    action: str                              # CORRECT | INCORRECT | AMBIGUOUS
    reason: str
    graded: list[GradedDoc] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    knowledge: str = ""


_INSUFFICIENT = "INSUFFICIENT"


def _looks_insufficient(text: str) -> bool:
    return text.strip().upper().startswith(_INSUFFICIENT)


def _assemble(sources: list[Source]) -> str:
    return "\n".join(f"[{i}] ({s.title}): {s.text}" for i, s in enumerate(sources, 1))


def _refined_sources(origin, hits_or_docs, query, chat_fn, model, max_sources):
    out: list[Source] = []
    for item in hits_or_docs[:max_sources]:
        doc_id = getattr(item, "doc_id", None) or item.doc.id
        title = getattr(item, "title", None) or item.doc.title
        text = getattr(item, "text", None) or item.doc.text
        refined = refine_doc(query, text, chat_fn=chat_fn, model=model)
        if refined:
            out.append(Source(origin, doc_id, title, refined))
    return out


def answer(
    query: str,
    internal: BM25Retriever,
    external: BM25Retriever | None = None,
    k: int = 4,
    max_sources: int = 3,
    model: str | None = None,
    chat_fn=chat,
) -> CRAGResult:
    """Run the full CRAG pipeline for ``query``.

    ``internal`` is the indexed knowledge base; ``external`` is the fallback source
    (a stand-in for the paper's web search). ``k`` is retrieval depth and
    ``max_sources`` caps how many documents get refined into the answer.
    """
    # 1. Retrieve from the internal index and 2. grade every hit.
    hits = internal.search(query, k=k)
    graded = [
        GradedDoc(h.doc.id, h.doc.title, h.doc.text,
                  grade(query, h.doc.text, chat_fn=chat_fn, model=model))
        for h in hits
    ]

    # 3. Decide the corrective action from the best relevance score.
    action = decide_action([g.score for g in graded])

    # 4. Assemble knowledge per the action.
    sources: list[Source] = []
    if action in (CORRECT, AMBIGUOUS):
        relevant = [g for g in graded if g.score >= 1]
        sources += _refined_sources("internal", relevant, query, chat_fn, model, max_sources)
    if action in (INCORRECT, AMBIGUOUS) and external is not None:
        ext_hits = external.search(query, k=k)
        sources += _refined_sources("external", ext_hits, query, chat_fn, model, max_sources)

    if not sources:
        return CRAGResult(query=query, answer=None, abstained=True, action=action,
                          reason="no usable knowledge after correction", graded=graded)

    # 5-6. Generate from the refined knowledge, or abstain if it isn't enough.
    knowledge = _assemble(sources)
    raw = chat_fn(GENERATE_PROMPT.format(question=query, knowledge=knowledge), model=model).strip()
    if _looks_insufficient(raw) or not raw:
        return CRAGResult(query=query, answer=None, abstained=True, action=action,
                          reason="assembled knowledge did not contain the answer",
                          graded=graded, sources=sources, knowledge=knowledge)

    return CRAGResult(query=query, answer=raw, abstained=False, action=action,
                      reason=f"answered via {action} using {len(sources)} refined source(s)",
                      graded=graded, sources=sources, knowledge=knowledge)


def default_internal() -> BM25Retriever:
    from .corpus import INTERNAL_CORPUS
    return BM25Retriever(INTERNAL_CORPUS)


def default_external() -> BM25Retriever:
    from .corpus import EXTERNAL_CORPUS
    return BM25Retriever(EXTERNAL_CORPUS)


__all__ = ["Source", "CRAGResult", "answer", "default_internal", "default_external"]
