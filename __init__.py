"""Corrective RAG (CRAG) — from scratch, dependency-free.

A retrieval-augmented generation pipeline that *grades its own retrieval* and
corrects it: it scores the retrieved documents, then either refines the good ones
(decompose-recompose), falls back to an external knowledge source when they're all
wrong, or combines both when unsure — before generating a grounded, cited answer.

Reference: Yan et al., 2024, "Corrective Retrieval-Augmented Generation",
https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

__version__ = "0.1.0"

from .corpus import (EXTERNAL_CORPUS, INTERNAL_CORPUS, Document, QAExample,
                     matches, normalize)
from .core import (CRAGResult, Source, answer, default_external, default_internal)
from .evaluator import (AMBIGUOUS, CORRECT, INCORRECT, GradedDoc, decide_action,
                        grade, parse_score)
from .refine import into_strips, refine_doc
from .retriever import BM25Retriever, Retrieved, tokenize

__all__ = [
    "__version__",
    "Document", "QAExample", "INTERNAL_CORPUS", "EXTERNAL_CORPUS", "matches", "normalize",
    "CRAGResult", "Source", "answer", "default_internal", "default_external",
    "CORRECT", "INCORRECT", "AMBIGUOUS", "GradedDoc", "decide_action", "grade", "parse_score",
    "into_strips", "refine_doc",
    "BM25Retriever", "Retrieved", "tokenize",
]
