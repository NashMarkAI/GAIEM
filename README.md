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