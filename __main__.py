"""CLI for Corrective RAG.

Usage:
    # Answer one question (internal index + external fallback)
    python -m corrective_rag "What is the largest planet in the Solar System?"

    # Show the corrective trace: grades, the action taken, and the refined sources
    python -m corrective_rag "Who wrote the play Hamlet?" --show-trace

    # Benchmark answer accuracy AND the corrective action on the bundled eval set
    python -m corrective_rag --bench
"""

from __future__ import annotations

import argparse
import sys

from .core import answer, default_external, default_internal
from .corpus import EVAL_QUESTIONS, matches
from .llm import DEFAULT_MODEL


def _print_trace(res) -> None:
    print("--- corrective trace ---")
    print("Internal retrieval graded (0-3):")
    if res.graded:
        for g in res.graded:
            print(f"  [{g.score}] {g.doc_id:<14} {g.title}")
    else:
        print("  (nothing retrieved internally)")
    print(f"Action: {res.action}")
    if res.sources:
        print("Refined knowledge used:")
        for i, s in enumerate(res.sources, 1):
            head = s.text if len(s.text) <= 70 else s.text[:67] + "..."
            print(f"  [{i}] ({s.origin}) {s.title}: {head}")
    print("------------------------")


def _answer_one(args, internal, external) -> int:
    res = answer(args.query, internal, external, k=args.k,
                 max_sources=args.max_sources, model=args.model)
    if args.show_trace:
        _print_trace(res)
    print("=" * 60)
    if res.abstained:
        print(f"I don't know — {res.reason}")
    else:
        print(res.answer)
        print(f"({res.reason})")
    return 0


def _bench(args, internal, external) -> int:
    ans_total = ans_ok = act_ok = 0
    for i, ex in enumerate(EVAL_QUESTIONS, 1):
        res = answer(ex.question, internal, external, k=args.k,
                     max_sources=args.max_sources, model=args.model)
        ans_total += 1
        correct = (not res.abstained) and matches(res.answer, ex)
        ans_ok += int(correct)
        action_correct = res.action == ex.expect_action
        act_ok += int(action_correct)
        verdict = "OK" if correct else ("ABSTAINED" if res.abstained else "WRONG")
        flag = "act:ok" if action_correct else f"act:{res.action}!={ex.expect_action}"
        shown = "(abstained)" if res.abstained else (res.answer or "")
        head = shown.replace("\n", " ")
        if len(head) > 40:
            head = head[:37] + "..."
        print(f"[{i:2d}] {verdict:<10} {flag:<22} gold={ex.answer:<16} {head}", flush=True)

    print("\n" + "=" * 64)
    print(f"Corrective RAG on bundled eval set — model={args.model or DEFAULT_MODEL}")
    print("=" * 64)
    print(f"  Answer accuracy:        {ans_ok}/{ans_total}")
    print(f"  Correct action chosen:  {act_ok}/{ans_total}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="corrective_rag",
        description="Corrective RAG (Yan et al., 2024): grade retrieval, then refine, "
                    "fall back to external knowledge, or combine — before generating.",
    )
    p.add_argument("query", nargs="?", help="The question to answer.")
    p.add_argument("--k", type=int, default=4, help="Passages to retrieve (default: 4).")
    p.add_argument("--max-sources", type=int, default=3,
                   help="Documents to refine into the answer (default: 3).")
    p.add_argument("--model", default=None, help=f"Model slug (default: {DEFAULT_MODEL}).")
    p.add_argument("--show-trace", action="store_true",
                   help="Print grades, the corrective action, and the refined sources.")
    p.add_argument("--bench", action="store_true",
                   help="Evaluate answer accuracy + corrective action on the bundled eval set.")
    args = p.parse_args()

    internal, external = default_internal(), default_external()

    if args.bench:
        return _bench(args, internal, external)
    if not args.query:
        p.error("provide a question to answer, or use --bench")

    print(f"\nQuestion: {args.query}", file=sys.stderr)
    print(f"Model: {args.model or DEFAULT_MODEL}\n", file=sys.stderr, flush=True)
    return _answer_one(args, internal, external)


if __name__ == "__main__":
    raise SystemExit(main())
