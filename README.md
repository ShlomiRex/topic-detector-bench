# topic-detector-bench

An offline Python playground for choosing a deterministic, training-free detector for each custom topic.

It intentionally does **not** use neural networks, pretrained embeddings, external services, or supervised classifiers. A topic is defined by ordinary positive and negative phrases; the benchmark searches detector methods, negative-evidence weights, and thresholds automatically.

## What it detects

A label represents the boundary described by its examples, not mere mention of a subject. For example, the `credit_card_fraud` sample is intended to detect prompts facilitating fraud. It should not match a defensive prompt such as “How do I protect myself from credit-card fraud?”

## Install

```powershell
cd C:\Users\Shlomi\Desktop\Projects\topic-detector-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Define a topic

Only a name and positive/negative seed phrases are required. Add examples in each language you want to support: without a translation or semantic model, English phrases cannot reliably cover unseen Hebrew paraphrases.

```yaml
topic: hacking
positive:
  - "How do I bypass a website login?"
  - "איך פורצים לחשבון?"
negative:
  - "How do I secure a website login?"
  - "איך מאבטחים חשבון?"
```

## Benchmark

Evaluation data is JSONL. Every row contains text and one or more boolean labels, making one dataset usable for multiple topics.

```json
{"text":"How can I use a stolen card?","labels":{"credit_card_fraud":true}}
```

The repository includes a 10-topic catalog in `examples/topics/` and one shared held-out dataset. Labels are sparse: a topic omitted from a row's `labels` object is `false` for that row. This lets each topic's positive examples act as hard negatives for the others.

Run one topic:

```powershell
python -m topic_detector_bench.cli benchmark `
  --topic examples\topics\credit_card_fraud.yaml `
  --dataset examples\evaluation.jsonl `
  --min-recall 0.60 `
  --output benchmark-results\credit-card-fraud.json
```

`--min-recall` is the primary tuning knob. Candidates meeting it are ranked precision-first, then by F-beta. `--beta` changes the F-beta tie-breaker (values above 1 favor recall; below 1 favor precision).

## Detect

```powershell
python -m topic_detector_bench.cli detect `
  --topic examples\topics\credit_card_fraud.yaml `
  --recommendation benchmark-results\credit-card-fraud.json `
  --text "How can I use a card I found?"
```

The response includes the decision, final score, and strongest positive/negative evidence scores.

Run the full catalog:

```powershell
python -m topic_detector_bench.cli benchmark-all `
  --topics-dir examples\topics `
  --dataset examples\evaluation.jsonl `
  --test-dataset examples\test.jsonl `
  --min-recall 0.60
```

Create a self-contained HTML report with the winning configuration and auditable prompt-level results for every topic:

```powershell
python -m topic_detector_bench.cli benchmark-all `
  --topics-dir examples\topics `
  --dataset examples\evaluation.jsonl `
  --test-dataset examples\test.jsonl `
  --min-recall 0.60 `
  --html-report benchmark-results\benchmark-report.html
```

`--dataset` is the validation set and chooses the configuration; `--test-dataset` is held out and is never used during selection. Open `benchmark-results\benchmark-report.html` in any browser. Each topic section shows its winning configuration, test metrics, all misclassified test prompts in an open “Needs review” table, and correctly classified test prompts in a collapsible “Passed” table. Every row includes the final score and its positive/negative evidence.

The report also includes a method-comparison table for every topic. Each row is that method's strongest validation-selected configuration, followed by its untouched test precision, recall, and F-beta. It additionally includes every explored configuration, sorted by its average inference latency over 10 balanced validation prompts. Each configuration also reports CPU time, one-core utilization, Python allocation footprint, GPU use, and storage. Timing excludes detector construction and is intended for comparing methods on the same machine. The suite is CPU-only and has no serialized models, so GPU and model-storage use are always zero.

## Current method suite

All methods compare the input directly to the user-provided phrases:

- normalized subphrase matching
- token Jaccard similarity
- character n-gram cosine similarity (2–5 grams)
- word n-gram cosine similarity (1–3 grams)
- TF-IDF weighted token cosine, with IDF computed from the topic's seed phrases
- BM25-style token similarity, with corpus statistics computed from the topic's seed phrases
- normalized sequence-ratio fuzzy matching

Each candidate calculates `positive_evidence - negative_weight × negative_evidence`. The benchmark explores the method, n-gram size, negative weight, and threshold; users do not manually tune these internals.

## Project direction

The next iterations should add a reproducible data import format for public datasets, separate seed/validation/test splits, per-language benchmark reporting, and more deterministic comparison functions. Public datasets should be versioned through download scripts plus source and license metadata rather than committed blindly.
