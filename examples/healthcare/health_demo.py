"""Healthcare variant of Corrective RAG.

A clinical assistant has *two* knowledge sources: the hospital's own indexed
**formulary** (internal) and a broader external reference (the paper's web search).
CRAG is a natural fit — grade what the formulary returned, and:

  * CORRECT   - the formulary covers it -> answer from refined internal knowledge.
  * INCORRECT - the formulary doesn't -> fall back to the external reference,
                instead of forcing an answer out of an irrelevant local page.
  * AMBIGUOUS - the formulary only grazes it -> combine internal + external.

This runs the real CRAG pipeline (corrective_rag.core.answer) over a small
formulary (health_internal.jsonl) and an external reference (health_external.jsonl).
It runs LIVE if a backend is configured, otherwise with a deterministic offline
critic that grades by token overlap and answers verbatim from the cited knowledge
(no fabricated medicine), so the whole corrective flow is demonstrable offline.

Run:  python -m corrective_rag.examples.healthcare.health_demo
  or: python examples/healthcare/health_demo.py   (from the repo root)

NOT MEDICAL ADVICE. Illustrative reference snippets only; each names its source. A
real clinical tool needs validated, versioned sources, a dense/hybrid retriever, a
trustworthy base model, PHI-safe inference, and a clinician in the loop.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# parents: [0] healthcare  [1] examples  [2] corrective_rag  [3] holds the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from corrective_rag.core import answer
from corrective_rag.corpus import Document
from corrective_rag.retriever import BM25Retriever, tokenize

HERE = Path(__file__).parent

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "in", "on",
    "and", "or", "what", "which", "how", "can", "should", "would", "do", "does",
    "with", "at", "be", "it", "that", "this", "as", "by", "from", "not", "no",
    "given", "give", "take", "safe", "during", "recovering", "patient", "patients",
}


def _terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def _field(prompt: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.+)", prompt)
    return m.group(1).strip() if m else ""


def _passage(prompt: str) -> str:
    return prompt.split('"""')[1] if '"""' in prompt else ""


def _knowledge(prompt: str) -> list[tuple[str, str]]:
    return re.findall(r"\[(\d+)\] \(.*?\): (.+)", prompt)


def extractive_critic(prompt: str, model: str | None = None) -> str:
    """Deterministic stand-in: grades by overlap, answers verbatim from knowledge."""
    if "retrieval evaluator" in prompt:
        overlap = len(_terms(_field(prompt, "Question")) & _terms(_passage(prompt)))
        return f"Score: {3 if overlap >= 2 else (1 if overlap == 1 else 0)}"
    if "using ONLY the knowledge below" in prompt:
        q = _terms(_field(prompt, "Question"))
        lines = _knowledge(prompt)
        if not lines:
            return "INSUFFICIENT"
        cite, text = max(lines, key=lambda kv: len(q & _terms(kv[1])))
        if not (q & _terms(text)):
            return "INSUFFICIENT"
        return f"{text} [{cite}]"
    return "?"


def load(path: Path) -> BM25Retriever:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            o = json.loads(line)
            docs.append(Document(id=o["id"], title=o.get("title", ""), text=o["text"]))
    return BM25Retriever(docs)


def pick_backend():
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("CRAG_BASE_URL"):
        from corrective_rag.llm import chat
        return chat, "LIVE (real LLM grading)"
    return extractive_critic, "OFFLINE (deterministic critic, no network)"


def show(res) -> None:
    print(f"\nQ: {res.query}")
    grades = ", ".join(f"{g.title}={g.score}" for g in res.graded) or "(nothing retrieved internally)"
    print(f"   internal grades: {grades}")
    print(f"   action: {res.action}")
    if res.sources:
        print("   sources: " + ", ".join(f"[{i}] {s.origin}:{s.title}"
                                          for i, s in enumerate(res.sources, 1)))
    if res.abstained:
        print(f"   -> ABSTAIN: {res.reason}")
    else:
        print(f"   -> {res.answer}")


QUESTIONS = [
    # covered by the formulary -> CORRECT (answer from internal)
    "What is the maximum daily dose of acetaminophen for adults?",
    # not in the formulary -> INCORRECT (fall back to the external reference)
    "Should aspirin be given to a child recovering from the flu?",
    # only grazed by the formulary -> AMBIGUOUS (combine internal + external)
    "Are ACE inhibitor blood pressure drugs safe during pregnancy?",
]


def main() -> int:
    chat_fn, label = pick_backend()
    internal = load(HERE / "health_internal.jsonl")
    external = load(HERE / "health_external.jsonl")
    print("=" * 74)
    print(f"Corrective RAG healthcare demo   |   mode: {label}")
    print(f"internal formulary: {len(internal.documents)} docs   "
          f"external reference: {len(external.documents)} docs")
    print("NOT MEDICAL ADVICE - illustrative demonstration of corrective retrieval.")
    print("=" * 74)
    for q in QUESTIONS:
        show(answer(q, internal, external, k=4, max_sources=3, chat_fn=chat_fn))
    print("\n" + "-" * 74)
    print("CRAG corrects retrieval instead of forcing an answer from the wrong page:")
    print("the aspirin question isn't in the formulary, so it falls back to the")
    print("external reference; the ACE question is only grazed internally, so it")
    print("combines both. Run with a real backend for live grading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
