"""Run a flight-evaluation corpus and save concise end-to-end benchmark results."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from flight_evaluation import FlightEvaluationState, PipelineEvaluation, evaluate


EXPERIMENT_DIR = Path(__file__).resolve().parent


class FlightEvaluationCase(BaseModel):
    id: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    state: FlightEvaluationState


class FlightEvaluationCorpus(BaseModel):
    test_states: list[FlightEvaluationCase] = Field(min_length=1)


def concise_result(evaluation: PipelineEvaluation) -> dict:
    """Keep the requested cost and performance values while retaining failures."""
    return {
        "end_to_end_ms": evaluation.total_ms,
        "input_tokens": evaluation.input_tokens,
        "output_tokens": evaluation.output_tokens,
        "total_token_cost_usd": evaluation.estimated_token_cost_usd,
        "error": evaluation.error or next(
            (metric.error for metric in evaluation.metrics if metric.error), None
        ),
    }


async def run_corpus(corpus: FlightEvaluationCorpus, output_path: Path | None = None) -> dict:
    """Evaluate cases sequentially so a case's timing is not affected by another."""
    results = []
    for case in corpus.test_states:
        print(f"Running test state {case.id}...", flush=True)
        try:
            report = await evaluate(case.state, case.prompt)
            results.append({
                "id": case.id,
                "jev": concise_result(report.jev),
                "legacy": concise_result(report.legacy),
            })
        except Exception as exc:  # A bad provider response must not erase prior runs.
            error = str(exc)
            results.append({
                "id": case.id,
                "jev": {"error": error},
                "legacy": {"error": error},
            })
        snapshot = {"results": results}
        if output_path is not None:
            output_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a corpus through both flight pipelines.")
    parser.add_argument(
        "--states", type=Path,
        default=EXPERIMENT_DIR / "flight_evaluation_state.json",
    )
    parser.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "results.json")
    parser.add_argument(
        "--id", type=int, dest="case_id",
        help="Run and merge only one test-state ID; useful for resumable live batches.",
    )
    args = parser.parse_args()
    load_dotenv()
    corpus = FlightEvaluationCorpus.model_validate_json(args.states.read_text())
    if args.case_id is not None:
        selected = [case for case in corpus.test_states if case.id == args.case_id]
        if not selected:
            parser.error(f"No test state has ID {args.case_id}.")
        corpus = FlightEvaluationCorpus(test_states=selected)

    previous = {item["id"]: item for item in json.loads(args.output.read_text()).get("results", [])} \
        if args.case_id is not None and args.output.exists() else {}
    current = asyncio.run(run_corpus(corpus, None))
    for item in current["results"]:
        previous[item["id"]] = item
    results = {"results": [previous[item_id] for item_id in sorted(previous)]}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {len(results['results'])} results to {args.output}")


if __name__ == "__main__":
    main()
