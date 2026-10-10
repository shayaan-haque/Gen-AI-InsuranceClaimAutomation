# Starter RAG evaluation dataset

`test_cases.json` contains 10 policy questions and reference answers checked against the synthetic ApexCare demonstration PDFs in `documents/`. These are not real insurance or regulatory advice.

## Fields

- `id`: stable identifier for comparing evaluation runs.
- `user_input`: question sent to the RAG system.
- `reference`: verified answer key, used only for evaluation. Never include it in the answer-generation prompt.
- `expected_clause_ids`: source clauses expected to support the answer. This is a custom field for clause-coverage checks, not a standard RAGAS field. Equivalent supporting clauses may exist, so this list is a starter target, not an exhaustive relevance annotation.

## Next step: collect actual outputs

For each case, run the question through your retriever and answer model, then collect:

```python
{
    "user_input": case["user_input"],
    "retrieved_contexts": [doc.page_content for doc in actual_context_docs],
    "response": generated_answer,
    "reference": case["reference"],
}
```

`retrieved_contexts` must contain the exact evidence passed to the answer model, including parent text if expanded. Do not fill it with the expected clauses or reference answer.

Keep `id` and `expected_clause_ids` separately for your own checks. The starter runner in `rag_eval/evals.py` maps `user_input` to `question` and `reference` to `grading_notes` for the RAGAS custom pass/fail metric. This metric checks answer correctness; retrieval precision, recall and faithfulness are not scored yet.

## Environment and running

Use the project-local Python environment, not the shared Anaconda installation. From the project root:

```bash
source .venv/bin/activate
python -m rag_eval.evals
```

Alternatively, run `.venv/bin/python -m rag_eval.evals` without activation. Ensure `.env` contains `GROQ_API_KEY`. A real run sends policy questions and evidence to Groq for answer generation and sends answers/reference notes for evaluation, consuming API quota.

To recreate the compatible environment using Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-eval.txt
```

Results are saved under `evaluation/experiments/` and exact answer-generation evidence is logged under `evaluation/logs/`. The adapter in `rag_eval/rag.py` uses the existing services for embeddings and hybrid retrieval; it evaluates policy Q&A rather than the complete invoice/claim workflow.

Start with TC-001 through TC-003 to learn the pipeline. TC-004 and TC-005 test rule/exception retrieval; TC-007 tests a calculation; TC-009 tests complete-list retrieval; TC-010 tests avoiding a blanket disease exclusion.

This is a small learning dataset, not a production benchmark. Later, add reviewed full-list, unanswerable, negated, ambiguous and end-to-end claim cases, plus human-validated decision labels.

Documentation: https://docs.ragas.io/en/stable/getstarted/rag_eval/
