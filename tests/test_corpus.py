"""The bundled corpora + eval set, and the answer-matching helper."""

import unittest

from corrective_rag.corpus import (EVAL_QUESTIONS, EXTERNAL_CORPUS, INTERNAL_CORPUS,
                                   matches, normalize)
from corrective_rag.evaluator import ACTIONS


class CorpusTests(unittest.TestCase):
    def test_corpora_nonempty_and_have_unique_ids(self):
        for corpus in (INTERNAL_CORPUS, EXTERNAL_CORPUS):
            self.assertTrue(corpus)
            ids = [d.id for d in corpus]
            self.assertEqual(len(ids), len(set(ids)))

    def test_eval_actions_are_valid(self):
        self.assertTrue(EVAL_QUESTIONS)
        for ex in EVAL_QUESTIONS:
            self.assertIn(ex.expect_action, ACTIONS)


class MatchTests(unittest.TestCase):
    def test_normalize_collapses_whitespace_and_case(self):
        self.assertEqual(normalize("  Hello   WORLD "), "hello world")

    def test_matches_substring(self):
        ex = EVAL_QUESTIONS[0]  # answer "Paris"? no -> "Jupiter"
        self.assertTrue(matches("The answer is Jupiter.", ex))
        self.assertFalse(matches("The answer is Saturn.", ex))

    def test_no_match_on_none(self):
        self.assertFalse(matches(None, EVAL_QUESTIONS[0]))


if __name__ == "__main__":
    unittest.main()
