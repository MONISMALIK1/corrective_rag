"""The LLM client's response handling — fully offline by patching urlopen."""

import json
import unittest
from unittest import mock

import corrective_rag.llm as llm


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ChatTests(unittest.TestCase):
    @mock.patch.object(llm, "_API_KEY", "test-key")
    def test_returns_assistant_content(self):
        payload = {"choices": [{"message": {"content": "hello world"}}]}
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)):
            self.assertEqual(llm.chat("hi"), "hello world")

    @mock.patch.object(llm, "_API_KEY", "test-key")
    def test_falls_back_to_reasoning_field(self):
        payload = {"choices": [{"message": {"reasoning": "because reasons"}}]}
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)):
            self.assertEqual(llm.chat("hi"), "because reasons")

    @mock.patch.object(llm, "_API_KEY", "test-key")
    def test_raises_on_inline_error(self):
        payload = {"error": {"message": "boom"}}
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)):
            with self.assertRaises(RuntimeError):
                llm.chat("hi")

    def test_requires_key_for_remote(self):
        with mock.patch.object(llm, "_API_KEY", ""), \
             mock.patch.object(llm, "_API_URL", "https://openrouter.ai/api/v1/chat/completions"):
            with self.assertRaises(EnvironmentError):
                llm.chat("hi")


if __name__ == "__main__":
    unittest.main()
