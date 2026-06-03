# corrective_rag

[![tests](https://github.com/MONISMALIK1/corrective_rag/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/corrective_rag/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **Corrective RAG (CRAG)** —
retrieval-augmented generation that **grades its own retrieval and corrects it**.
It scores the retrieved documents, then either *refines* the good ones, *falls back*
to an external source when they're all wrong, or *combines* both when unsure — before
generating a grounded, cited answer.

> Yan, Gu, Zhu, Ling, Cai, Zhang (2024), *Corrective Retrieval-Augmented Generation.*
> [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

## Why this exists — picking up where Self-RAG leaves off

[Self-RAG](https://github.com/MONISMALIK1/self_rag) reflects on whether retrieved
passages are *relevant* and *supported*, and **abstains** when they look weak. But if
retrieval simply pulls the **wrong** documents, reflection can only make the system
give up — it can't *fix* the retrieval. CRAG adds the missing corrective layer:

| | Self-RAG | **Corrective RAG** |
| --- | --- | --- |
| Judges retrieved docs | yes (relevant? supported?) | yes (a 0–3 relevance grade) |
| When retrieval is bad | **abstains** | **corrects**: refine / fall back / combine |
| Knowledge cleanup | uses whole passages | **decompose–recompose** into knowledge strips |
| External fallback | — | yes (web search; here, a second corpus) |

## Faithful technique, pragmatic adaptation

Two honest substitutions, in the spirit of the rest of this series:

- The paper trains a lightweight **T5 retrieval evaluator**; we don't train a model,
  so we elicit the same relevance signal by **prompting** and parse a single 0–3 score.
- The paper's fallback is large-scale **web search**; offline, we model "the web" as a
  second in-memory corpus (`EXTERNAL_CORPUS`). The *control flow* is identical — only
  the knowledge source behind the fallback is swapped — so the whole thing runs and is
  tested with zero network.

## The pipeline

```
retrieve from the internal index (BM25)
   │
grade every document  ── retrieval evaluator, 0–3
   │
decide the action from the best score:
   ├─ CORRECT   → refine the relevant internal docs (decompose–recompose)
   ├─ INCORRECT → discard them; fall back to the external knowledge source
   └─ AMBIGUOUS → combine refined internal knowledge with external
   │
assemble the refined knowledge (with citations)
   │
generate a grounded, cited answer  ── or abstain if nothing usable remains
```

**Knowledge refinement** decomposes each document into sentence-level *strips*, grades
every strip, drops the irrelevant ones, and recomposes the survivors — so the generator
sees concentrated knowledge instead of noisy passages.

## Install

No third-party dependencies. Python 3.11+.

```bash
git clone https://github.com/MONISMALIK1/corrective_rag.git
cd corrective_rag && pip install -e .      # optional; or just run from the parent dir
```

Point it at any OpenAI-compatible backend:

```bash
# OpenRouter (default)
export OPENROUTER_API_KEY=sk-or-...

# …or a local model — no key, no cloud
export CRAG_BASE_URL=http://localhost:11434/v1/chat/completions   # Ollama
export CRAG_MODEL=qwen2.5:7b
```

## Use

```bash
# well covered internally -> CORRECT (refine internal knowledge)
python -m corrective_rag "What is the largest planet in the Solar System?"

# absent internally -> INCORRECT (fall back to the external source) — show the trace
python -m corrective_rag "Who wrote the play Hamlet?" --show-trace

# thinly covered internally -> AMBIGUOUS (combine internal + external)
python -m corrective_rag "How long has the storm on the largest planet lasted?" --show-trace

# accuracy AND whether the right corrective action was chosen, on the eval set
python -m corrective_rag --bench
```

Example trace:

```
--- corrective trace ---
Internal retrieval graded (0-3):
  [0] jupiter        Jupiter
  [0] saturn         Saturn
Action: INCORRECT
Refined knowledge used:
  [1] (external) William Shakespeare: William Shakespeare was an English playwright who wrote...
------------------------
============================================================
Shakespeare wrote the play Hamlet around 1600. [1]
(answered via INCORRECT using 1 refined source(s))
```

## Design

Everything that *decides* is pure stdlib and unit-tested offline; only the grading,
refinement, and generation calls touch the network.

| Module | Responsibility |
| --- | --- |
| `retriever.py` | from-scratch **BM25** over an in-memory corpus (pure, deterministic) |
| `evaluator.py` | the **retrieval evaluator** — grade parsing + the Correct/Incorrect/Ambiguous decision |
| `refine.py` | **decompose–recompose** knowledge refinement (strip filtering) |
| `prompts.py` | the grade + grounded-generation prompts |
| `core.py` | the corrective control flow → `CRAGResult` |
| `corpus.py` | internal + external ("web") corpora and an eval set covering all three actions |
| `llm.py` | backend-agnostic OpenAI-compatible client (OpenRouter or local) |

## Test

```bash
make test        # or: python -m unittest discover -s corrective_rag/tests -t . -v
```

26 offline tests, no API key required — covering BM25 ranking, score parsing, the
action decision, strip-level refinement, the full control flow across all three
corrective actions, and the abstention path, all driven by a scripted fake LLM.

## Limitations

- **Evaluator quality is the base model's, not learned.** The 0–3 grade comes from
  prompting rather than a trained evaluator, so it's only as good as the model's
  zero-shot judgement.
- **"External" is a stand-in.** Offline it's a second corpus; wiring it to a real web
  search is a drop-in swap of the external retriever — the corrective logic is unchanged.
- **BM25 is lexical.** Retrieval matches words, not meaning; a dense retriever is a
  drop-in replacement on either index.
- **Cost scales with strips.** Grading every document and every strip costs model
  calls; `--max-sources` caps the documents refined into the answer.

## Part of a series

From-scratch, dependency-free reimplementations of LLM-reasoning papers:
[self_rag](https://github.com/MONISMALIK1/self_rag) ·
[react_agent](https://github.com/MONISMALIK1/react_agent) ·
[reflexion](https://github.com/MONISMALIK1/reflexion) ·
[pal](https://github.com/MONISMALIK1/pal) ·
[self_consistency](https://github.com/MONISMALIK1/self_consistency) ·
[self_discover](https://github.com/MONISMALIK1/self_discover) ·
[tree_of_thoughts](https://github.com/MONISMALIK1/tree_of_thoughts) ·
**corrective_rag** (this repo).

## License

MIT
