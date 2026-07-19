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

## 🧪 What the Baseline Measures

The frozen GAIEM v0.1.0 baseline contains four independent nine-turn conversations, producing thirty-six model responses under identical observation states and evaluator conditions.

Those four conversations examine three broader real-world risk areas.

### 🩺 1. Clinical State Retention and Safety — `TRIAGE-CHAT-009`

The clinical conversation tests whether a model can preserve and reconstruct a progressively developing patient account.

The tested state includes:

- chest pain and chest tightness Safety — `TRIAGE-CHAT-009`

The clinical conversation tests whether a model can preserve and reconstruct a progressively;
- shortness of breath;
- dizziness;
- symptom onset and progression;
- worsening symptoms during exertion;
- warfarin use;
- previous blood-clot history;
- penicillin allergy;
- pauses while speaking;
- patient uncertainty and anxiety;
- incidental or distracting information introduced later in the conversation.

The purpose is not to test whether the model can produce a diagnosis. It is to test whether clinically material information remains present, accurate and correctly separated from uncertainty and distraction as the conversation develops.

This matters because failure in a clinical interaction is not limited to hallucinating a diagnosis. A system may also create risk by forgetting medication, losing an allergy, changing the chronology, omitting a worsening symptom or failing to reconstruct the complete patient state when asked.

The clinical benchmark is therefore relevant to clinicians, clinical-system designers, healthcare AI developers, auditors and organisations evaluating conversational systems for safety-sensitive use.

### 📚 2. Source and Factual Integrity

Two benchmark conversations examine information integrity from different directions.

`SOURCE-CHAT-009` tests whether the model:

- distinguishes supplied source material from unsupported claims;
- maintains uncertainty where the evidence is incomplete;
- avoids inventing citations, authors, findings or conclusions;
- preserves source restrictions across later turns;
- resists pressure to convert an unsupported claim into a factual statement.

`FACT-CHAT-009` tests whether the model:

- retains named project facts;
- preserves ownership, dates, deadlines and responsibilities;
- reconstructs cumulative project state;
- avoids replacing established facts with plausible alternatives;
- recalls the complete record during explicit later probes.

These sessions test whether the model can maintain an evidence boundary and preserve factual state rather than merely generating a plausible response.

### 📋 3. Instruction Persistence and Behavioural Stability — `INSTRUCTION-CHAT-009`

The instruction-retention conversation tests whether the model continues to follow user-defined requirements as the exchange becomes longer.

It examines whether the model can preserve:

- required output format;
- prohibited content;
- wording constraints;
- ordering rules;
- scope restrictions;
- instructions established several turns earlier;
- later corrections that supersede earlier instructions.

This measures whether the model remains operationally aligned with the continuing task instead of following only the most recent prompt.

## 🩺 Clinical Dataset Provenance

The clinical-language work is grounded in the **MTS-Dialog dataset**, published with the EACL 2023 research paper *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*.

MTS-Dialog contains approximately 1,700 short doctor-patient conversations with corresponding clinical section summaries. Its main dataset includes 1,201 training conversations, 100 validation conversations and two 200-conversation test sets, including material used in the MEDIQA-Chat and MEDIQA-Sum 2023 challenges. ([GitHub][1])

GAIEM currently uses:

`MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv`

The source dataset is not inserted directly into the benchmark without review. GAIEM contains a separate clinical-data preparation pipeline.

### `clinical_data_loader.py`

The loader:

- reads the MTS-Dialog training CSV;
- preserves the source file, source row and source identifier;
- records the source-file integrity digest;
- parses clinician and patient turns;
- separates accepted records from malformed or rejected records;
- preserves the original wording;
- does not modify the source dataset;
- does not generate diagnoses or call an external model.

### `clinical_language_curator.py`

The curator maps patient language into controlled clinical concepts, including:

- chest pain;
- shortness of breath;
- dizziness;
- anticoagulant use;
- allergy;
- exertion;
- onset;
- progression;
- indigestion or heartburn.

It also distinguishes:

- positive symptom statements;
- symptoms explicitly denied;
- uncertain statements;
- short answers whose meaning depends on the preceding clinician question;
- acute wording from historical wording.

### `clinical_candidate_selector.py`

The selector ranks audited conversations for the chest-pain triage scenario using transparent clinical-language and red-flag criteria.

The mapped terms include:

- chest pain, pressure and tightness;
- breathlessness and difficulty breathing;
- dizziness;
- warfarin and anticoagulant use;
- worsening with walking or exertion;
- allergy language;
- indigestion and heartburn.

The selector does not diagnose, rewrite source language or automatically convert a patient record into a benchmark case.

## 🧩 How the Clinical Benchmark Was Constructed

The GAIEM triage conversation uses **original composite patient language informed by the audited and curated MTS-Dialog material**.

It does not reproduce one identifiable patient conversation as the benchmark.

The construction process was:

**MTS-Dialog source records**  
→ audited clinician/patient turn parsing  
→ accepted and rejected record separation  
→ clinical-language concept mapping  
→ chest-pain and red-flag candidate ranking  
→ reviewed vocabulary and conversational patterns  
→ original controlled nine-turn GAIEM triage conversation  
→ deterministic expected facts and observation-state testing

This gives the clinical scenario real conversational relevance while retaining a controlled benchmark state against which factual retention, omission, drift, uncertainty and hallucination can be measured.

The MTS-Dialog repository also contains augmented datasets and a correlation-study dataset with expert fact-based scoring of generated clinical summaries. The current GAIEM v0.1.0 clinical preparation pipeline directly uses the main training dataset; the augmented and correlation-study files are retained as separate source resources and must not be described as part of the measured baseline unless they are explicitly incorporated into a later benchmark version.

---


## 🧩 Pluggable Domain and Dataset Architecture

GAIEM is not restricted to healthcare, clinical triage or the datasets used in the frozen v0.1.0 baseline. The clinical benchmark demonstrates the method, but the framework is designed so that different datasets, professional domains, legal frameworks, regulatory materials and controlled source collections can be connected to the same evaluation architecture.

The underlying process remains consistent:

**domain dataset or controlled source material**  
→ source auditing and preparation  
→ terminology, concept and relationship mapping  
→ controlled multi-turn benchmark construction  
→ deterministic expected-state definition  
→ model execution  
→ drift, hallucination, factuality, instruction and retention evaluation  
→ preserved evidence, charts and reporting

### ⚖️ Legal and Regulatory Evaluation

A legal implementation could connect GAIEM to an authorised and version-controlled dataset containing:

- legislation and statutory provisions;
- procedural rules;
- regulatory frameworks;
- judgments and case-law extracts;
- contractual clauses;
- legal definitions;
- pleaded facts and evidence;
- case chronologies;
- party identities and procedural positions;
- identified duties, rights and alleged breaches.

A controlled legal benchmark could then test whether a model:

- preserves the correct legal framework throughout a continuing case discussion;
- distinguishes established facts, disputed allegations, evidence and legal conclusions;
- retains dates, parties, procedural stages and pleaded positions accurately;
- identifies relevant duties or potential breaches without inventing provisions;
- avoids fabricating authorities, quotations, judgments or citations;
- maintains jurisdictional and procedural boundaries;
- recognises when the supplied material is insufficient to support a conclusion;
- avoids changing the meaning of legislation or case evidence through generalised narration;
- preserves corrections and later instructions that supersede earlier information;
- reconstructs the complete legal state when explicitly asked.

This would allow GAIEM to expose whether a model begins accurately but later drifts from the legislation, changes the facts, merges allegations with evidence, invents legal authority or produces confident narration unsupported by the controlled record.

### 🔌 Configurable Benchmark Components

GAIEM operates as a configurable evaluation framework rather than a fixed collection of questions.

Users can configure or replace:

- the source dataset;
q- the professional or regulatory domain;
- benchmark conversations;
- the number of sessions;
- the number of turns within each session;
- observation states and recall probes;
- expected facts and protected terms;
- instructions and prohibited behaviours;
- factual aliases and equivalent expressions;
- domain-specific evaluators;
- pass thresholds;
- evidence requirements;
- reporting and chart outputs;
- local or external model providers;
- model versions, quantisation and inference settings.

A benchmark may contain nine turns, ninety turns or a substantially longer controlled conversation. Observation states can be positioned at fixed intervals, after important facts are introduced, following corrections or at points where contextual interference is expected.

### 🌐 Expansion into Other Domains

The same architecture can be applied to:

- healthcare and clinical safety;
- law and legal procedure;
- regulatory compliance;
- public administration;
- finance and accounting;
- engineering and technical assurance;
- scientific research;
- customer service;
- education;
- insurance;
- procurement;
- safeguarding;
- internal organisational policy.

Each domain can introduce its own controlled sources, terminology maps, factual relationships, risk conditions and evaluator rules while retaining the same evidence-preserving GAIEM execution process.

### 📊 What GAIEM Reveals

GAIEM does not assume that every model will drift or hallucinate in every individual response.

It creates a controlled and repeatable method for revealing:

- whether a failure occurs;
- when it first appears;
- which fact, instruction or source boundary is lost;
- whether the failure is omission, contradiction, invention or semantic drift;
- whether the failure worsens as the conversation grows;
- whether the model recovers after correction;
- whether the same failure repeats across model versions or providers.

The framework therefore shows what a model is actually doing across a continuing conversation rather than relying on a capability claim, model card or isolated demonstration.

### 🧱 Versioning and Comparability

Material changes to benchmark content, source datasets, evaluator behaviour, domain configuration or expected-state definitions must be released under a new benchmark or framework version.

The frozen GAIEM v0.1.0 baseline remains unchanged.

Healthcare, legal, regulatory and other domain-specific benchmark packages can therefore be developed as separate, versioned evaluation profiles while remaining comparable through the same controlled execution, evidence-preservation and reporting architecture.

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

[1]: https://github.com/abachaa/MTS-Dialog "GitHub - abachaa/MTS-Dialog: A collection of doctor-patient conversations and corresponding clinical notes and summaries."
