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

## Submission Notes

-   Public app link enables interactive testing
-   Includes reproducible test cases
-   Ready for extension with embedding models or LLM judges

------------------------------------------------------------------------

## Author

**Bhumika Khatwani**\
BeyondChats -- LLM Evaluation Pipeline
