"""Lance le banc d'évaluation sur les trois variantes et écrit un rapport Markdown + JSON.

Usage :
    python -m eval.run_eval                     # toutes les variantes
    python -m eval.run_eval --variants hybrid_rerank --min-accuracy 0.6
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from docqa.agent import answer
from docqa.config import settings
from docqa.rag import VARIANTS

from .metrics import abstained, citation_correct, citation_precision, faithful, is_correct, recall_at_k

HERE = Path(__file__).parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(variant: str, questions: list[dict]) -> dict:
    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        res = answer(q["question"], variant)
        latencies.append((time.perf_counter() - t0) * 1000)
        row = {"id": q["id"], "type": q["type"], "route": res.route, "answer": res.answer, "citations": res.citations}
        if q["type"] == "unanswerable":
            row["hallucinated"] = not abstained(res.answer)
        else:
            row["correct"] = is_correct(res.answer, q["facts"])
            row["cite_ok"] = citation_correct(res.citations, q["gold"])
            row["cite_prec"] = citation_precision(res.citations, q["gold"])
            if q["type"] == "rag" and res.route == "rag":
                row["recall"] = recall_at_k(res.passages, q["gold"])
                row["faithful"] = faithful(res.answer, res.citations, res.passages)
        rows.append(row)

    def rate(key, subset=None):
        vals = [r[key] for r in rows if key in r and r[key] is not None and (subset is None or r["type"] == subset)]
        return round(sum(vals) / len(vals), 3) if vals else None

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    precs = [r["cite_prec"] for r in answerable if r.get("cite_prec") is not None]
    return {
        "variant": variant,
        "n_questions": len(rows),
        "accuracy": rate("correct"),
        "recall_at_k": rate("recall"),
        "citation_accuracy": rate("cite_ok"),
        "citation_precision": round(statistics.mean(precs), 3) if precs else None,
        "faithfulness": rate("faithful"),
        "hallucination_rate": rate("hallucinated"),
        "routing_accuracy": round(
            sum(1 for r in answerable if (r["route"] == "sql") == (r["type"] == "sql")) / len(answerable), 3
        ),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "rows": rows,
    }


def to_markdown(results: list[dict]) -> str:
    mode = f"LLM `{settings.llm_model}`" if settings.llm_api_key else "extractif (sans LLM)"
    cols = [
        ("accuracy", "Exactitude"),
        ("recall_at_k", f"Recall@{settings.top_k}"),
        ("citation_accuracy", "Citations exactes"),
        ("faithfulness", "Fidélité"),
        ("hallucination_rate", "Hallucination ↓"),
        ("routing_accuracy", "Routage"),
        ("latency_p50_ms", "Latence p50 (ms)"),
    ]
    lines = [
        f"# Résultats d'évaluation\n\nGénération : {mode} · Embeddings : `{settings.embedding_model}` · "
        f"Reranker : `{settings.reranker_model}` · {results[0]['n_questions']} questions\n",
        "| Variante | " + " | ".join(c[1] for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for r in results:
        cells = [
            (
                "—"
                if r[k] is None
                else (f"{r[k]:.0%}" if isinstance(r[k], float) and k != "latency_p50_ms" else str(r[k]))
            )
            for k, _ in cols
        ]
        lines.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--min-accuracy", type=float, default=None, help="échoue si la meilleure variante est en dessous")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    results = [evaluate(v, questions) for v in args.variants]
    args.out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(results)
    (args.out / "results.md").write_text(md, encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    if args.min_accuracy is not None:
        best = max(r["accuracy"] or 0 for r in results)
        if best < args.min_accuracy:
            raise SystemExit(f"Exactitude {best:.0%} < seuil {args.min_accuracy:.0%}")


if __name__ == "__main__":
    main()
