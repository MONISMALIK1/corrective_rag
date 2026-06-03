"""A from-scratch BM25 retriever — pure Python, zero dependencies, deterministic.

CRAG is about *correcting* retrieval, not about the retriever itself, so this stays
deliberately simple and inspectable: classic Okapi BM25 (Robertson & Walker, 1994)
over an in-memory corpus. No embeddings, no vector index, no network — retrieval is
reproducible and the whole pipeline is unit-tested offline.

The same retriever indexes both the internal corpus and the external ("web")
corpus; CRAG decides *which* knowledge to trust, the retriever just ranks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Document

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. The one place text becomes terms."""
    return _TOKEN.findall(text.lower())


@dataclass
class Retrieved:
    """One search hit: the document, its BM25 score, and 1-based rank."""
    doc: Document
    score: float
    rank: int


class BM25Retriever:
    """Okapi BM25 over a fixed list of :class:`~corrective_rag.corpus.Document`."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        df: Counter[str] = Counter()
        for doc in self.documents:
            tokens = tokenize(f"{doc.title} {doc.text}")
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            df.update(tf.keys())

        n = len(self.documents)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _score(self, query_terms: list[str], i: int) -> float:
        tf = self._tfs[i]
        dl = self._lengths[i]
        denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            total += self._idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_norm)
        return total

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        """Return the top-``k`` documents that match ``query``, best first.

        Only positively scored documents are returned, so an off-topic query
        yields an empty list — itself a signal the internal index is a poor match.
        """
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scored = [(self._score(terms, i), i) for i in range(len(self.documents))]
        scored.sort(key=lambda s: (-s[0], s[1]))

        hits: list[Retrieved] = []
        for score, i in scored[:k]:
            if score <= 0.0:
                break
            hits.append(Retrieved(doc=self.documents[i], score=round(score, 4), rank=len(hits) + 1))
        return hits


__all__ = ["BM25Retriever", "Retrieved", "tokenize"]
