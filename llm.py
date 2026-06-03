"""LLM wrapper — a single ``chat`` call over the OpenAI-compatible API.

Same proven, dependency-free pattern as the rest of this series: HTTP over stdlib
``urllib``, the API key read only from the environment and never logged or written
to disk.

Backend-agnostic by design: the payload is the OpenAI-compatible
``/chat/completions`` shape, so anything that speaks it works. OpenRouter is the
default, but point ``CRAG_BASE_URL`` at a **local** server — Ollama, LM Studio,
llama.cpp's server, vLLM — and no cloud (and no API key) is needed::

    export CRAG_BASE_URL=http://localhost:11434/v1/chat/completions   # Ollama
    export CRAG_MODEL=qwen2.5:7b

CRAG uses the model both as a *retrieval evaluator* and as a generator. The grading
calls want a single, low-variance verdict, so ``chat`` defaults to greedy
(temperature 0).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("CRAG_MODEL", "openai/gpt-oss-120b:free")
_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("CRAG_API_KEY", "")
_API_URL = os.environ.get("CRAG_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
_MAX_RETRIES = 5
_RETRY_BASE = 2.0

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CTRL.sub("", text)


def _is_local(url: str) -> bool:
    return any(host in url for host in ("localhost", "127.0.0.1", "0.0.0.0"))


def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 768,
    stop: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Single OpenAI-compatible chat completion. Returns the assistant text."""
    if not _API_KEY and not _is_local(_API_URL):
        raise EnvironmentError(
            "No API key set. For OpenRouter: export OPENROUTER_API_KEY=sk-or-...  "
            "For a local model: export CRAG_BASE_URL=http://localhost:11434/v1/chat/completions "
            "(then no key is needed)."
        )

    payload: dict = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    if "openrouter" in _API_URL:
        headers["HTTP-Referer"] = "https://github.com/MONISMALIK1/corrective_rag"
        headers["X-Title"] = "Corrective RAG: grade, refine, and correct retrieval"

    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(_API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = _clean(resp.read().decode())
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After", _RETRY_BASE ** attempt))
                time.sleep(wait)
                continue
            raise

        if "error" in data and "choices" not in data:
            raise RuntimeError(f"API error: {data['error']}")

        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return text.strip()

    raise RuntimeError("Exceeded max retries calling the chat API")


__all__ = ["DEFAULT_MODEL", "chat"]
