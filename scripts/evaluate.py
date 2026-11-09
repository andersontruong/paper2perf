"""Evaluate original-document cases; annotations are never passed to the agent.

Run from the repository root: python scripts/evaluate.py cases.json --condition agentic
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from paper2perf.config import Settings
from paper2perf.documents import extract_pages
from paper2perf.schemas import ChatRequest
from paper2perf.runner import InvestigationRunner
from paper2perf.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--condition", choices=["agentic", "one-shot"], default="agentic")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.condition == "one-shot":
        settings = replace(settings, max_retrievals=1, max_repairs=0)
    if not settings.model or not settings.api_key:
        parser.error("Configure the model and API key before evaluation")
    store = Store(settings.data_dir)
    runner = InvestigationRunner(settings, store)
    summaries = []
    for case in json.loads(args.cases.read_text()):
        sid = store.new_session()
        documents = []
        for document in case["documents"]:
            source = args.cases.parent / document
            raw = source.read_bytes()
            suffix, pages = extract_pages(source.name, raw, settings.max_pages)
            documents.append(store.add_document(sid, source.name, suffix, raw, pages))
        request = ChatRequest(message=case["question"], mode=case.get("mode", "simulate"),
                              document_ids=documents,
                              assumptions=case.get("assumptions", [])).model_dump()
        # Only the explicit request and originals enter the agent. Gold annotations do not.
        start = time.monotonic()
        run = runner.run(sid, request)
        rid = run["id"]
        result = run["result"] or {}
        predicted = (result.get("evidence") or {}).get("facts", [])
        gold = case.get("expected_facts", [])
        matched = 0
        for expected in gold:
            matched += any(
                f["parameter"] == expected["parameter"] and f["unit"] == expected["unit"]
                and f["variant"] == expected["variant"]
                and math.isclose(f["value"], expected["value"], rel_tol=1e-6)
                for f in predicted
            )
        summaries.append({"case": case["id"], "condition": args.condition, "run_id": rid,
                          "status": run["status"], "elapsed_seconds": time.monotonic() - start,
                          "simulation_completed": bool(result.get("metrics")),
                          "makespan_seconds": (result.get("metrics") or {}).get("makespan_seconds"),
                          "model_calls": len(list((store.root / "runs" / rid).glob("llm-*-request.json"))),
                          "extracted_fact_count": len(predicted), "annotated_fact_count": len(gold),
                          "matched_annotated_facts": matched,
                          "annotated_fact_recall": matched / len(gold) if gold else None,
                          "note": "Exact annotation matching; semantic citation entailment needs manual review"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
