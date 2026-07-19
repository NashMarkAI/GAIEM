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

- Ollama — implemented as the local model-execution provider.
  - `llama3.2:latest` — empirically tested and retained as the frozen GAIEM v0.1.0 baseline.
  - `gemma4:12b` — installed as a local Ollama model and scheduled as the next empirical evaluation.
- OpenAI — adapter file present in the source; not empirically tested in GAIEM v0.1.0.
- DeepSeek — planned provider adapter; not implemented or empirically tested.

## Local Deployment Principle

GAIEM is designed to evaluate models that can be executed locally as well as models exposed through external provider APIs.

The planned `gemma4:12b` evaluation will run as a local Ollama instance on a laptop with an Intel i9 processor and 16 GB RAM. The purpose of the test is to establish whether the model and the complete evidence-preserving GAIEM workflow can operate locally on that hardware.

For the Gemma evaluation, local execution means:

- model inference is performed on the local laptop through Ollama;
- no external model API is used for the benchmark run;
- prompts and model responses remain within the locally controlled environment;
- the same frozen benchmark can be repeated without dependence on a remote model endpoint;
- raw responses, transcripts, evaluator records, score tables, manifests, charts and reports remain under the operator’s control;
- model execution, evidence capture, evaluation and report generation are performed within the same locally controlled workflow.

The result will establish the measured capability and limitations of this specific local configuration. It will not be treated as evidence that every model, model size or workload can operate efficiently on equivalent hardware.

## Planned Development

The following work is planned after the frozen GAIEM v0.1.0 Llama 3.2 baseline. The existing v0.1.0 benchmark, evaluator implementations and baseline evidence will not be retrospectively altered.

### Gemma4:12b empirical evaluation

- [ ] Run `gemma4:12b` as a local Ollama instance on a laptop with an Intel i9 processor and 16 GB RAM.
- [ ] Use the unchanged GAIEM v0.1.0 runner.
- [ ] Use the unchanged `chat_unscaffolded.json` benchmark.
- [ ] Use the same evaluator versions used for the Llama 3.2 baseline.
- [ ] Preserve the raw provider responses, transcripts, evaluator records, score tables, manifests and generated charts.
- [ ] Publish the results as a separate Gemma4:12b model-evaluation supplement.
- [ ] Maintain the Gemma work on a separate comparison branch.
- [ ] Compare its model-quality results with the frozen `llama3.2:latest` baseline.
- [ ] Keep hardware-dependent latency and throughput results clearly identified as measurements from the local i9 and 16 GB execution environment.

### BERTScore semantic evaluator

- [ ] Add BERTScore as a separate semantic-analysis evaluator.
- [ ] Use it to detect cases where a model preserves the meaning of a required fact through different wording.
- [ ] Address known strict-matcher undercounting involving equivalent expressions such as:
  - “short of breath” and “feels unable to get enough air”;
  - “the pain has not subsided” and “the pain has not properly gone away”;
  - “Priya owns Project Cedar” and “Project Cedar is owned by Priya”.
- [ ] Keep the existing deterministic token-based evaluator unchanged.
- [ ] Report strict literal coverage and BERTScore semantic coverage as separate measurements.
- [ ] Do not silently merge the two measurements into one score.
- [ ] Introduce BERTScore under a new evaluator and framework version.
- [ ] Rerun the Llama 3.2 baseline under the new version before making cross-model comparisons.

BERTScore is being added because literal token matching is deterministic and auditable but can classify a valid paraphrase as a missing fact. BERTScore adds contextual meaning-level comparison. It will supplement the strict evaluator rather than replace it, allowing GAIEM to show both literal retention and semantic retention.

### Benchmark alias enrichment

- [ ] Add controlled aliases for verified equivalent expressions identified during manual audit.
- [ ] Document each alias and the reason it is considered equivalent.
- [ ] Keep the existing v0.1.0 benchmark permanently unchanged.
- [ ] Publish the enriched benchmark under a new benchmark version.
- [ ] Rerun the baseline before using the enriched benchmark for later model comparisons.

### DeepSeek provider adapter

- [ ] Implement a dedicated DeepSeek provider adapter.
- [ ] Validate request construction, model identification, response storage, token reporting and error handling.
- [ ] Keep DeepSeek marked as planned until the adapter exists.
- [ ] Do not describe DeepSeek as supported or empirically tested until a completed evidence-preserving run has been recorded.
- [ ] Record any external API cost separately from model-quality results.

### Bayesian predictive analyser

- [ ] Develop the predictive analyser after empirical results from multiple models are available.
- [ ] Use completed GAIEM runs to estimate likely performance for models that have not yet been tested.
- [ ] Keep forecasts clearly separate from measured empirical results.
- [ ] Report uncertainty intervals and source confidence.
- [ ] Leave unknown proprietary model specifications unknown rather than inventing values.
- [ ] Update the predictive model as each new empirical evaluation is completed.
- [ ] Use information-gain analysis to identify which model would provide the most valuable next test.

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