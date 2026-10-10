import os
import sys
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
import json
import argparse
import asyncio
import httpx
from collections import Counter
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
load_dotenv()
# Allow both `python -m rag_eval.evals` and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rag_eval.rag import rag_client
from rag_eval.rate_limit import pace_async_request, REQUEST_INTERVAL_SECONDS

client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    max_retries=2,
    http_client=httpx.AsyncClient(event_hooks={"request": [pace_async_request]}),
)

llm = llm_factory(
    "qwen/qwen3.8-27b",
    provider="openai",
    client=client,
    max_tokens=800,
    temperature=0,
)

context_precision = ContextPrecision(llm=llm)
context_recall = ContextRecall(llm=llm)
faithfulness = Faithfulness(llm=llm)

def load_dataset(limit=1):
    dataset = Dataset(
        name="test_cases",
        backend="local/csv",
        root_dir=str(PROJECT_ROOT / "evaluation"),
    )

    with open(PROJECT_ROOT / "evaluation" / "test_cases.json", encoding="utf-8") as file:
        data_samples = json.load(file)

    for sample in data_samples[:limit]:
        row = {
            "id": sample["id"],
            "question": sample["user_input"],
            "grading_notes": sample["reference"],
            "expected_clause_ids": sample.get("expected_clause_ids", []),
        }
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


_case_lock = asyncio.Semaphore(1)


async def _score_case(row):
    # Fixed fields allow failed and successful cases to coexist in the CSV.
    result = {
        **row,
        "response": "",
        "retrieved_contexts": [],
        "retrieved_child_contexts": [],
        "score": None,
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "log_file": "",
        "status": "failed",
        "errors": {},
    }
    try:
        response = rag_client.query(row["question"])
    except Exception as error:
        result["errors"]["generation"] = str(error)
        return result

    result.update({
        "response": response["answer"],
        "retrieved_contexts": response["retrieved_contexts"],
        "retrieved_child_contexts": response["retrieved_child_contexts"],
        "log_file": response.get("logs", ""),
    })
    metric_calls = [
        ("context_precision", context_precision, {
            "user_input": row["question"],
            "reference": row["grading_notes"],
            "retrieved_contexts": response["retrieved_child_contexts"],
        }),
        ("context_recall", context_recall, {
            "user_input": row["question"],
            "reference": row["grading_notes"],
            "retrieved_contexts": response["retrieved_child_contexts"],
        }),
        ("faithfulness", faithfulness, {
            "user_input": row["question"],
            "response": response["answer"],
            "retrieved_contexts": response["retrieved_contexts"],
        }),
        ("score", my_metric, {
            "llm": llm,
            "response": response["answer"],
            "grading_notes": row["grading_notes"],
        }),
    ]
    for name, metric, arguments in metric_calls:
        try:
            result[name] = (await metric.ascore(**arguments)).value
        except Exception as error:
            # Keep the answer and other metric scores; don't manufacture a zero.
            result["errors"][name] = str(error)
            print(f"{row['id']}: {name} failed; error recorded.")

    result["status"] = "partial" if result["errors"] else "success"
    return result


@experiment()
async def run_experiment(row):
    # RAGAS schedules rows concurrently; limit this workflow to one at a time.
    async with _case_lock:
        return await _score_case(row)


async def main(limit=1):
    dataset = load_dataset(limit)
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    statuses = Counter(row["status"] for row in experiment_results)
    print(f"Experiment finished: {dict(statuses)}")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = PROJECT_ROOT / "evaluation" / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate insurance policy Q&A with RAGAS.")
    parser.add_argument("--limit", type=int, default=1, help="Number of cases (default: 1).")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    print(f"API requests are paced {REQUEST_INTERVAL_SECONDS:g}s apart; scoring can take several minutes per case.")
    asyncio.run(main(args.limit))
