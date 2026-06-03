"""Knowledge refinement: strip splitting + keeping only relevant strips."""

import unittest

from corrective_rag.refine import into_strips, refine_doc


class IntoStripsTests(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self):
        strips = into_strips("Jupiter is large. It has a storm! Is it a star? No.")
        self.assertEqual(strips,
                         ["Jupiter is large.", "It has a storm!", "Is it a star?", "No."])

    def test_empty(self):
        self.assertEqual(into_strips("   "), [])


class RefineDocTests(unittest.TestCase):
    def _grader(self, relevant_substr):
        # Score a passage 3 if it contains the keyword, else 0.
        def fake(prompt, model=None):
            passage = prompt.split('"""')[1] if '"""' in prompt else prompt
            return "Score: 3" if relevant_substr in passage.lower() else "Score: 0"
        return fake

    def test_drops_irrelevant_strips(self):
        text = "Jupiter is the largest planet. Bananas are yellow fruit."
        refined = refine_doc("largest planet", text, chat_fn=self._grader("planet"))
        self.assertIn("Jupiter", refined)
        self.assertNotIn("Bananas", refined)

    def test_falls_back_to_best_strip_when_all_filtered(self):
        text = "Alpha statement here. Beta statement there."
        # Nothing contains the keyword -> all score 0 -> keep the single best (first).
        refined = refine_doc("nonexistent", text, chat_fn=self._grader("zzz"))
        self.assertTrue(refined)  # never empty for a non-empty doc
        self.assertIn("statement", refined)


if __name__ == "__main__":
    unittest.main()
