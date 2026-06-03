"""The retrieval evaluator's pure parts: score parsing and the action decision."""

import unittest

from corrective_rag.evaluator import (AMBIGUOUS, CORRECT, INCORRECT, decide_action,
                                      parse_score)


class ParseScoreTests(unittest.TestCase):
    def test_reads_last_line(self):
        self.assertEqual(parse_score("It is relevant.\nScore: 3"), 3)

    def test_reads_inline(self):
        self.assertEqual(parse_score("Score: 2 because it mentions the topic"), 2)

    def test_clamps_to_range(self):
        self.assertEqual(parse_score("Score: 9"), 3)
        self.assertEqual(parse_score("Score: -4"), 0)

    def test_defaults_to_zero_when_unparseable(self):
        self.assertEqual(parse_score("I cannot tell."), 0)

    def test_prefers_final_score_line(self):
        self.assertEqual(parse_score("Score: 0 maybe\nFinal Score: 3"), 3)


class DecideActionTests(unittest.TestCase):
    def test_correct_when_a_doc_is_clearly_relevant(self):
        self.assertEqual(decide_action([3, 0, 1]), CORRECT)
        self.assertEqual(decide_action([2]), CORRECT)

    def test_incorrect_when_all_irrelevant(self):
        self.assertEqual(decide_action([0, 0, 0]), INCORRECT)
        self.assertEqual(decide_action([]), INCORRECT)  # nothing retrieved

    def test_ambiguous_in_between(self):
        self.assertEqual(decide_action([1, 0]), AMBIGUOUS)
        self.assertEqual(decide_action([1]), AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
