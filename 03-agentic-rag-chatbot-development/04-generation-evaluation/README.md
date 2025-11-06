# Generation Evaluation

This directory contains notebooks used to evaluate and improve the Canvas Assistant's response generation capabilities through experiments and automated testing.

## Answer Relevancy Metric

The Answer Relevancy metric uses LLM-as-a-judge to evaluate the quality of the RAG pipeline's responses. It measures how relevant the generated output is compared to the user's input query. Using DeepEval's GPTModel with the o4-mini model, the metric provides both a score and a detailed explanation for the assessment.

## Notebooks

- `00-naive-generation-evaluation.ipynb`: Initial baseline evaluation using an LLM-based answer relevancy metric. The notebook achieved a 23% pass rate across 100 test cases.

- `01-improving-naive-generation.ipynb`: Implementation of advanced RAG techniques including:
  - Query rewriting: Using LLMs to break down and optimize search queries
  - Document filtering: LLM-based contextual compression to extract relevant information
  - Tool-augmented responses: An agentic workflow where the LLM has the ability to autonomously make multiple tool calls until desired output is produced
  
  The notebook achieved a 37% pass rate across 100 test cases, showing improvement over the baseline.

## Implementation Details

The evaluation uses:
- `AnswerRelevancyMetric` from the DeepEval framework with o4-mini model as judge
- Claude model (via Amazon Bedrock) for generation
- Chroma vector store with BGE embeddings for retrieval
- LangGraph for the agentic workflow implementation

## Test Results

- Baseline (00-naive-generation): 23% pass rate (23/100 test cases)
- Improved (01-improving-naive-generation): 37% pass rate (37/100 test cases)

Test cases were sourced from synthetic datasets and evaluated using automated relevancy metrics.