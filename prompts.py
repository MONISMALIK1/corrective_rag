"""Prompts for CRAG's three model-driven steps: grade, refine, generate.

The paper trains a lightweight T5 *retrieval evaluator*; we don't have that model,
so we elicit the same relevance signal by prompting and parse a single integer
score out of each reply — the same prompting-instead-of-finetuning adaptation the
rest of this series uses. Every prompt asks for its value on the **last line** so
parsing is robust even when the model explains itself first.
"""

# Retrieval evaluator: score how relevant one passage (a whole doc, or a single
# knowledge strip during refinement) is to the question.
RELEVANCE_SCORE_PROMPT = """You are a retrieval evaluator. Judge how relevant the passage is to \
answering the question — only whether it contains useful information, ignoring style.

Question: {question}

Passage:
\"\"\"{passage}\"\"\"

Respond on the last line with exactly: Score: N
where N is an integer from 0 (completely irrelevant) to 3 (directly answers it)."""

# Grounded generation from the assembled (refined) knowledge.
GENERATE_PROMPT = """Answer the question using ONLY the knowledge below. Keep it to one sentence.
If the knowledge does not actually contain the answer, reply with exactly: INSUFFICIENT

Question: {question}

Knowledge:
{knowledge}

End your answer by citing the source number(s) in square brackets, e.g. [1].
Answer:"""

__all__ = ["RELEVANCE_SCORE_PROMPT", "GENERATE_PROMPT"]
