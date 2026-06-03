"""The CRAG control flow end to end, with real BM25 + a scripted evaluator/generator.

Drives all three corrective actions offline:
  * CORRECT   - relevant internal docs -> refined internal knowledge.
  * INCORRECT - irrelevant internal docs -> fall back to external knowledge.
  * AMBIGUOUS - thin internal hit -> combine internal + external.
plus the abstention path when generation yields INSUFFICIENT.
"""

import re
import unittest

from corrective_rag.core import answer
from corrective_rag.corpus import Document
from corrective_rag.evaluator import AMBIGUOUS, CORRECT, INCORRECT
from corrective_rag.retriever import BM25Retriever

_STOP = {"the", "is", "a", "an", "of", "to", "in", "on", "and", "or", "what", "who",
         "how", "has", "for", "it", "its", "are", "was", "were", "this", "that"}


def _terms(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _question(prompt):
    m = re.search(r"Question:\s*(.+)", prompt)
    return m.group(1) if m else ""


def _passage(prompt):
    return prompt.split('"""')[1] if '"""' in prompt else ""


def make_critic(answer_text="The answer is grounded in the sources. [1]"):
    """Grades by question/passage token overlap; generates a fixed answer."""

    def critic(prompt, model=None):
        if "retrieval evaluator" in prompt:
            overlap = len(_terms(_question(prompt)) & _terms(_passage(prompt)))
            score = 3 if overlap >= 2 else (1 if overlap == 1 else 0)
            return f"Score: {score}"
        if "using ONLY the knowledge below" in prompt:
            return answer_text
        return "?"

    return critic


class CorrectActionTests(unittest.TestCase):
    def test_uses_refined_internal_knowledge(self):
        internal = BM25Retriever([
            Document("jup", "Jupiter", "Jupiter is the largest planet in the Solar System."),
        ])
        external = BM25Retriever([
            Document("x", "X", "Totally unrelated external content about cooking."),
        ])
        res = answer("What is the largest planet?", internal, external, chat_fn=make_critic())
        self.assertEqual(res.action, CORRECT)
        self.assertFalse(res.abstained)
        self.assertTrue(res.sources)
        self.assertTrue(all(s.origin == "internal" for s in res.sources))


class IncorrectActionTests(unittest.TestCase):
    def test_falls_back_to_external(self):
        internal = BM25Retriever([
            Document("jup", "Jupiter", "Jupiter is the largest planet in the Solar System."),
        ])
        external = BM25Retriever([
            Document("shake", "Shakespeare", "Shakespeare wrote the play Hamlet around 1600."),
        ])
        res = answer("Who wrote the play Hamlet?", internal, external, chat_fn=make_critic())
        self.assertEqual(res.action, INCORRECT)
        self.assertFalse(res.abstained)
        self.assertTrue(res.sources)
        self.assertTrue(all(s.origin == "external" for s in res.sources))

    def test_abstains_if_external_also_empty(self):
        internal = BM25Retriever([Document("jup", "Jupiter", "Jupiter is a planet.")])
        external = BM25Retriever([Document("c", "C", "Cooking recipes and kitchen tips.")])
        res = answer("Who wrote Hamlet?", internal, external, chat_fn=make_critic())
        self.assertEqual(res.action, INCORRECT)
        self.assertTrue(res.abstained)


class AmbiguousActionTests(unittest.TestCase):
    def test_combines_internal_and_external(self):
        internal = BM25Retriever([
            Document("jup", "Jupiter", "Jupiter has a famous storm."),
        ])
        external = BM25Retriever([
            Document("grs", "Storm", "The storm has lasted for centuries on the planet."),
        ])
        res = answer("How long has the storm lasted?", internal, external, chat_fn=make_critic())
        self.assertEqual(res.action, AMBIGUOUS)
        self.assertFalse(res.abstained)
        origins = {s.origin for s in res.sources}
        self.assertIn("internal", origins)
        self.assertIn("external", origins)


class AbstentionTests(unittest.TestCase):
    def test_abstains_when_generation_insufficient(self):
        internal = BM25Retriever([
            Document("jup", "Jupiter", "Jupiter is the largest planet in the Solar System."),
        ])
        external = BM25Retriever([Document("x", "X", "Unrelated content.")])
        res = answer("What is the largest planet?", internal, external,
                     chat_fn=make_critic(answer_text="INSUFFICIENT"))
        self.assertEqual(res.action, CORRECT)
        self.assertTrue(res.abstained)
        self.assertIn("did not contain", res.reason)


if __name__ == "__main__":
    unittest.main()
