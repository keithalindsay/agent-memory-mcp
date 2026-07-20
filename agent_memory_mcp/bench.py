"""Eval harness: graph-vs-vector recall on a labeled seed set (DESIGN.md §9, §11 step 13).

Ingests the seed corpus through the REAL remember() pipeline, runs each labeled
question through graph-only and vector-only recall, grades, prints a Rich
scorecard, and writes results/scorecard.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .embedders import get_embedder
from .memory import Memory
from .models import GradedResult, Question, Scorecard, load_questions, read_corpus_lines

_PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = _PKG_ROOT / "seed" / "corpus.txt"
DEFAULT_QUESTIONS = _PKG_ROOT / "seed" / "questions.json"
CATEGORIES = ["single_hop", "multi_hop", "aggregation"]
METHODS = ["vector", "graph"]


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def grade(predicted: Optional[str], gold: str) -> bool:
    """Loose containment match on normalized text."""
    p, g = _norm(predicted), _norm(gold)
    if not p or not g:
        return False
    return g in p or p in g


def run_eval(
    corpus_lines: list[str],
    questions: list[Question],
    embedder_name: str = "hash",
    dim: int = 256,
) -> Scorecard:
    """Ingest the corpus and evaluate both backends. Returns a Scorecard."""
    mem = Memory(":memory:", embedder=get_embedder(embedder_name, dim=dim), dim=dim)
    for line in corpus_lines:
        mem.remember(line)

    results: list[GradedResult] = []
    for q in questions:
        for method in METHODS:
            res = mem.recall(q.text, method=method)
            results.append(
                GradedResult(
                    question_id=q.id,
                    method=method,
                    category=q.category,
                    predicted=res.answer,
                    gold=q.gold_answer,
                    correct=grade(res.answer, q.gold_answer),
                )
            )
    return _build_scorecard(results)


def _build_scorecard(results: list[GradedResult]) -> Scorecard:
    by_mc: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    overall: dict[str, float] = {}
    for method in METHODS:
        for cat in CATEGORIES:
            rows = [r for r in results if r.method == method and r.category == cat]
            by_mc[method][cat] = round(sum(r.correct for r in rows) / len(rows), 3) if rows else 0.0
        mrows = [r for r in results if r.method == method]
        overall[method] = round(sum(r.correct for r in mrows) / len(mrows), 3) if mrows else 0.0
    n = len({r.question_id for r in results})
    return Scorecard(by_method_category=by_mc, overall=overall, n=n, results=results)


def render(scorecard: Scorecard, console: Optional[Console] = None) -> None:
    console = console or Console()
    table = Table(title="Recall accuracy by category")
    table.add_column("method", justify="left", style="bold")
    for cat in CATEGORIES:
        table.add_column(cat, justify="right")
    table.add_column("overall", justify="right", style="bold")
    for method in METHODS:
        row = [method]
        for cat in CATEGORIES:
            row.append(f"{scorecard.by_method_category[method][cat]:.2f}")
        row.append(f"{scorecard.overall[method]:.2f}")
        table.add_row(*row)
    console.print(table)


def run_bench(
    corpus_path: Optional[str] = None,
    questions_path: Optional[str] = None,
    db: str = ":memory:",
    embedder: Optional[str] = None,
    out_path: str = "results/scorecard.json",
) -> Scorecard:
    """CLI entry point: run the benchmark, print the scorecard, write JSON."""
    corpus = read_corpus_lines(corpus_path or DEFAULT_CORPUS)
    questions = load_questions(questions_path or DEFAULT_QUESTIONS)
    scorecard = run_eval(corpus, questions, embedder_name=embedder or "hash")

    render(scorecard)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard.model_dump(), indent=2), encoding="utf-8")
    Console().print(f"Wrote {out}")
    return scorecard
