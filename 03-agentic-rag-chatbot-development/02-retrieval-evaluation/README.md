# Retrieval Evaluation

This directory contains the implementation and results of evaluating and improving the retrieval mechanisms used in the Canvas Assistant chatbot.

# Retrieval evaluation

This directory contains the notebooks, results and analysis used to evaluate and improve the retrieval pipeline that feeds the Canvas Assistant. The work here focuses on measuring retrieval quality, experimenting with retrieval-time transformations, and making product decisions based on empirical results.

## What’s in this folder
- `00-naive-retriever-evaluation.ipynb`: baseline evaluation and initial metrics collection.
- `01-improving-naive-retrieving.ipynb`: experiments with query-rewriting, contextual compression and evaluation harnesses.
- `data/eval_results/`: saved evaluation runs and artifacts.

## Metrics and key results

We measure retrieval quality primarily with a Contextual Relevancy metric (threshold 0.7, evaluated with an LLM evaluator).

Contextual Relevancy
- Definition: the Contextual Relevancy metric uses an LLM-as-a-judge to measure the quality of a retriever by evaluating how relevant the information presented in the retrieval context is for a given input. It is a self-explaining LLM-eval that returns both a score and a reason for that score.

Important results from the experiments:
- Contextual Relevancy (pass rate) improved from 5.26% to 36.84% after introducing query-rewriting and contextual compression — roughly a 7x relative improvement on the test set.

## Techniques evaluated and product decisions

Query-rewriting
- Purpose: rewrite or break multi-part user messages into focused search queries to improve retrieval precision.
- Effectiveness: produced substantial evaluation gains in the experiments.
- Multiple experiments and manual inspection of chat logs show that the LLM often internally filters and reformulates noisy user messages into single focused questions. But custom query-rewriting is utilized for consistent reformulation of queries.

Contextual compression
- Purpose: compress retrieved contexts to the minimal relevant information before passing them to the LLM. This improves relevancy and reduces tokens consumed by the LLM.

Important caveat: both query-rewriting and contextual compression add extra processing steps and therefore increase end-to-end retrieval latency. The trade-off between accuracy/cost and latency should be considered when deciding whether and where to enable them in production.

## Outputs

- Evaluation artifacts and run logs are stored under `../data/eval_results/`.