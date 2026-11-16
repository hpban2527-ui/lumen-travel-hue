"""Run the six Practice 03 questions against a configured Ollama server."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from guide_loader import load_guide
from rag import answer_question, clean_base_url, embed_texts, RAGError

ROOT = Path(__file__).resolve().parent
CASES = [
    ("Answerable", "What is the Imperial City in Huế?", {"G01"}, None),
    ("Answerable", "Tell me about Dong Ba Market.", {"G03"}, None),
    ("Answerable", "Which river flows through Huế?", {"G05"}, None),
    ("Paraphrase", "Where can I learn about Nguyen royal court history?", {"G01", "G11", "G16"}, None),
    ("Outside guide", "What is the ticket price today at the Imperial City?", set(), False),
    ("Outside guide", "Which tombs will be open at 9 p.m. tomorrow?", set(), False),
]


def assess(result, expected_ids, expected_support):
    retrieved = {m.entry["id"] for m in result.matches}
    if expected_support is False:
        return (not result.supported, "generation" if result.supported else "—")
    if not retrieved.intersection(expected_ids):
        return False, "retrieval"
    if not result.supported:
        return False, "generation"
    citations = {s for s in expected_ids if f"[{s}]" in result.text}
    if not citations:
        return False, "generation"
    return True, "—"


def main():
    parser = argparse.ArgumentParser(description="Measure the six Assignment 03 RAG questions")
    parser.add_argument("--out", default="evaluation_results.json", help="JSON report path")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    try:
        base = clean_base_url(os.getenv("OLLAMA_BASE_URL", ""))
        guide = load_guide()
        texts = [f"{e['title']} | {e['category']} | {' '.join(e['tags'])} | {e['text']}" for e in guide]
        vectors = embed_texts(base, texts)
    except (RAGError, OSError, ValueError) as exc:
        parser.exit(2, f"Cannot start evaluation: {exc}\n")
    rows = []
    for kind, question, ids, expected_support in CASES:
        try:
            result = answer_question(base, question, guide, vectors, k=3)
            correct, failure = assess(result, ids, expected_support)
            rows.append({"type": kind, "question": question,
                         "expected_ids": sorted(ids) if ids else ["I don't know"],
                         "retrieved": [m.entry["id"] for m in result.matches],
                         "scores": [round(m.score, 5) for m in result.matches],
                         "answer": result.text, "supported": result.supported,
                         "correct": correct, "failure_stage": failure})
        except RAGError as exc:
            rows.append({"type": kind, "question": question,
                         "expected_ids": sorted(ids) if ids else ["I don't know"],
                         "retrieved": [], "scores": [], "answer": "",
                         "supported": None, "correct": False,
                         "failure_stage": "integration", "error": str(exc)})
    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "embedding_model": "embeddinggemma", "chat_model": "qwen3:4b",
              "question_count": len(rows), "passed": sum(r['correct'] for r in rows),
              "results": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{report['passed']}/{len(rows)} questions met the evaluation rules\n")
    print("| Type | Question | Expected | Retrieved (scores) | Correct | Failure stage |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        hits = ", ".join(f"{i} ({s:.3f})" for i, s in zip(r['retrieved'], r['scores'])) or "—"
        print(f"| {r['type']} | {r['question']} | {', '.join(r['expected_ids'])} | {hits} | {'YES' if r['correct'] else 'NO'} | {r['failure_stage']} |")
    print(f"\nSaved {out}")
    return 0 if report['passed'] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
