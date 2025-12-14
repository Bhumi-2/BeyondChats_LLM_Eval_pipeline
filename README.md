# BeyondChats -- LLM Response Evaluation Pipeline

This project implements a lightweight, real-time **LLM evaluation
pipeline** to assess AI-generated responses against retrieved context.

It evaluates AI answers on: - Response relevance & completeness -
Hallucination / factual accuracy - Latency & cost estimation

The solution includes: - A CLI-based evaluation script
(`evaluator.py`) - An interactive Streamlit dashboard (`app.py`) - A
public demo link with built-in test cases

------------------------------------------------------------------------

## Live Demo (Public Streamlit App)

A **publicly accessible Streamlit app** is provided to demonstrate
real-time evaluation with predefined and custom test cases.

**Live App URL:**\
`https://beyondchatsllmevalpipeline-34aqkqqzxtdiphpnrxn64m.streamlit.app`

The app includes: - Auto-fill demo test cases (good answer, hallucinated
answer, incomplete answer) - Adjustable thresholds - JSON report
download

------------------------------------------------------------------------

## Features

### Evaluation Metrics

-   **Relevance Score** -- semantic overlap with retrieved context
-   **Completeness Score** -- coverage of user intent
-   **Lexical Support Score** -- sentence-level factual grounding
-   **Numeric Hallucination Detection**
-   **Latency Measurement**
-   **Token & Cost Estimation**

All metrics are heuristic-based and API-free, enabling fast and
deterministic execution.

------------------------------------------------------------------------

## Project Structure

    BeyondChats_LLM_Eval_Submission/
    ├── evaluator.py
    ├── app.py
    ├── sample-chat-conversation-01.json
    ├── sample-chat-conversation-02.json
    ├── sample_context_vectors-01.json
    ├── sample_context_vectors-02.json
    ├── README.md

------------------------------------------------------------------------

## Included Test Scenarios

The Streamlit app includes **built-in test cases**:

1.  **Good Answer**
    -   High relevance and completeness
    -   No hallucinations
2.  **Hallucinated Numeric Answer**
    -   Unsupported numeric claims flagged
    -   Lower factual support score
3.  **Incomplete Answer**
    -   Correct facts but missing details
    -   Lower completeness score

These allow reviewers to quickly validate evaluator behavior.

------------------------------------------------------------------------

## How to Run Locally

### CLI Evaluation

``` bash
python3 evaluator.py   --chat sample-chat-conversation-01.json   --context sample_context_vectors-01.json   --out report.json
```

### Streamlit Dashboard

``` bash
python3 -m streamlit run app.py
```

------------------------------------------------------------------------

## Output

The evaluation output includes: - Per-turn scores - Flagged unsupported
sentences - Unsupported numeric claims - Latency and estimated token
cost - Downloadable JSON report

------------------------------------------------------------------------

## Design Notes

-   No external LLM APIs are used
-   Metrics are modular and production-pluggable
-   Designed for real-time evaluation pipelines

------------------------------------------------------------------------

## Why This Design?

This evaluation pipeline is intentionally built using lightweight, heuristic-based metrics instead of LLM-based judges or embedding APIs.

Key design considerations:

Real-time performance:
LLM-as-a-judge approaches introduce high latency (hundreds of milliseconds to seconds per evaluation). This solution uses only local text processing, enabling millisecond-level evaluations suitable for real-time systems.

Cost efficiency at scale:
External API-based evaluation (LLM judges, embedding services) incurs significant costs when applied to millions of daily conversations. This pipeline performs all evaluations locally, ensuring near-zero marginal cost per evaluation.

Deterministic and explainable scoring:
Heuristic metrics such as lexical overlap, sentence-level support, and numeric validation provide transparent, interpretable results. This avoids the non-determinism and opacity often associated with LLM-based judges.

Production extensibility:
The architecture is modular by design. Individual metrics (e.g., relevance or factual support) can be replaced with embedding-based similarity or LLM judges in the future without changing the overall pipeline.

Operational simplicity:
The solution requires no GPU, no external services, and no network calls, making it easy to deploy, scale horizontally, and integrate into existing AI systems.

This design prioritizes speed, cost control, and reliability, making it suitable for large-scale, real-time AI response evaluation while remaining extensible for more advanced evaluation methods in the future.

------------------------------------------------------------------------

## Submission Notes

-   Public app link enables interactive testing
-   Includes reproducible test cases
-   Ready for extension with embedding models or LLM judges

------------------------------------------------------------------------

## Author

**Bhumika Khatwani**\
BeyondChats -- LLM Evaluation Pipeline
