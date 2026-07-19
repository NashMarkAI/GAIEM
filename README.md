# GenAI Evaluation Matrix (GAIEM)

## Overview

The GenAI Evaluation Matrix is an open-source Python framework for evaluating
Generative AI systems using deterministic and repeatable benchmark testing.

Rather than relying on subjective impressions, the framework measures
behaviour using reproducible metrics across multiple domains.

Current evaluation modules include:

- Accuracy
- Hallucination Detection
- Consistency
- Goal Drift
- Execution Drift
- Loop Detection
- Recovery
- Reliability

The framework has been designed to support any Generative AI model that
provides either an API or a local inference endpoint.

---

## Provider and Model Status

- Ollama — implemented
  - `llama3.2:latest` — empirically tested baseline
  - `gemma4:12b` — next planned empirical test on the second MacBook
- OpenAI — adapter file present in the source; not empirically tested in GAIEM v0.1.0
- DeepSeek — planned provider adapter; not implemented or empirically tested

## Planned Development

The following work is planned after the frozen GAIEM v0.1.0 Llama 3.2 baseline.

- [ ] **Gemma4:12b empirical evaluation**
  - Run `gemma4:12b` through the unchanged v0.1.0 benchmark, runner and evaluator set.
  - Record the second MacBook hardware and Ollama configuration.
  - Publish the results as a separate model-evaluation supplement and comparison branch.

- [ ] **BERTScore semantic evaluator**
  - Add BERTScore as an additional semantic-analysis evaluator.
  - The current strict token-based evaluators can undercount valid recall when a model preserves meaning through paraphrasing.
  - BERTScore will compare contextual semantic similarity and help identify equivalent expressions that do not share the same literal wording.
  - It will supplement, not replace, the deterministic strict evaluator so that both literal coverage and semantic coverage remain visible.
  - This change will require a new framework version and a rerun of the baseline before cross-model comparisons are made.

- [ ] **Benchmark alias enrichment**
  - Add controlled semantic aliases for known equivalent expressions identified during the manual audit.
  - Preserve the existing v0.1.0 benchmark unchanged.
  - Publish the enriched benchmark under a new benchmark version.

- [ ] **DeepSeek provider adapter**
  - Implement a dedicated DeepSeek provider adapter.
  - Validate request handling, response evidence, model identification and error reporting before any empirical test.
  - Keep DeepSeek marked as planned until the adapter exists and a completed run has been recorded.

- [ ] **Bayesian predictive analyser**
  - Develop a separate forecasting layer after results from multiple models are available.
  - Use empirical GAIEM results to estimate likely performance for untested models.
  - Report uncertainty intervals rather than presenting forecasts as measured results.
  - Update the predictive model as each new empirical model run is completed.

## Installation

Clone the repository.

```bash
git clone <repository-url>

cd genai-evaluation-matrix
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Copy:

```text
.env.example
```

to

```text
.env
```

and insert your own API credentials.

---

## Run

```bash
python runner.py
```

---

## Project Status

Current Version

**0.1 Alpha**

This project is under active development.