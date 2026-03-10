# Twelve Labs Model Evaluation: Worker Safety Video Classification

Independent benchmark of **Marengo 3.0** (embeddings) and **Pegasus 1.2** (generation) on industrial worker safety video classification.

## Key Findings

| Benchmark | Metric | Score |
|---|---|---|
| Marengo Embeddings | kNN 8-class accuracy | **0.499** |
| Marengo Embeddings | kNN binary (safe/unsafe) | **0.640** |
| Marengo Embeddings | Silhouette score | **0.024** |
| Marengo Retrieval | mAP@5 | **0.426** |
| Marengo Retrieval | Recall@5 | **0.195** |
| Pegasus Generation | 8-class accuracy | **0.200** |
| Pegasus Generation | Format compliance | **0.400** |
| Pegasus Generation | Hallucination rate | **0.600** |

### What works

- **Object-level retrieval** is strong — "forklift safely transporting" returns AP@5=1.0
- **Visually distinct categories** like "Safe Carrying" reach 67% recall from Pegasus
- 512-dim embeddings are compact and practical for downstream ML pipelines

### What fails

- **State discrimination** — Marengo cannot distinguish open vs closed panels, inside vs outside walkways, authorized vs unauthorized workers (AP=0.0 on all state-based queries)
- **Pegasus defaults to "Unsafe"** in 57% of responses — a binary heuristic, not 8-class understanding
- **Critical safety misclassification** — "Safe Walkway Violation" labeled as "Safe Walkway" in 2/3 cases
- **60% hallucination rate** despite constrained prompting at temperature=0.1

## Dataset

[Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) (CC-BY-4.0)

- 691 videos across 8 classes (4 safe, 4 unsafe)
- 1920×1080, 24fps, 1–20 seconds
- Evaluation subset: 72 sampled from test split, 50 successfully indexed

| Safe | Unsafe |
|---|---|
| Safe Walkway | Safe Walkway Violation |
| Authorized Intervention | Unauthorized Intervention |
| Closed Panel Cover | Opened Panel Cover |
| Safe Carrying | Carrying Overload with Forklift |

## Benchmarks

### A. Embedding Quality (Marengo 3.0)

Evaluates whether 512-dim embeddings separate safety classes in vector space.

- **kNN with cross-validation** (k=5, cosine distance, StratifiedKFold)
- **Silhouette score** for cluster cohesion

### B. Retrieval Quality (Marengo 3.0)

Text-to-video search with 10 natural language queries mapped to expected ground-truth labels.

```
"electrical panel left open and unattended"    → AP=0.83
"forklift safely transporting a normal load"   → AP=1.00
"person walking outside the safe walkway"      → AP=0.00
"panel cover properly closed and secured"      → AP=0.00
```

### C. Generation Quality (Pegasus 1.2)

Zero-shot classification with a constrained prompt. Measures exact-match accuracy, format compliance, and hallucination rate.

Pegasus output distribution (30 responses):
```
"Unsafe"        17  (not a valid label — hallucinated)
"Safe Walkway"   8
"Safe Carrying"  4
"SAFE"           1  (wrong casing — hallucinated)
```

## Quickstart

```bash
git clone <this-repo>
cd 12labs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TWELVE_LABS_API_KEY=<your_key>

# Quick sanity check (24 videos, ~5 min)
python eval_harness.py --split test --samples-per-label 3

# Full evaluation (72+ videos, ~20 min with rate limit pauses)
python eval_harness.py --split test --samples-per-label 10
```

First run downloads ~9.5GB from HuggingFace. State persists to `eval_state.json` so interrupted runs resume without re-indexing.

## Rate Limits

The free tier allows 8 requests/minute across all endpoints. The harness implements proactive throttling (10s intervals) and retry-after parsing, but expect ~20% failure rate on large batches. This is itself a finding — batch evaluation workflows are impractical at the free tier.

## Files

| File | Description |
|---|---|
| `eval_harness.py` | Evaluation harness — dataset loading, indexing, benchmarks |
| `eval_results.json` | Raw per-sample results (video IDs, predictions, embedding dims) |
| `experiment_report.json` | Full experiment report with methodology, per-class breakdowns, and analysis |
| `requirements.txt` | Python dependencies |

## Reproducibility

- Python 3.13
- `twelvelabs` SDK 1.2.0
- Random seed: 42
- Results may vary slightly due to Pegasus non-determinism at temperature=0.1

See `experiment_report.json` for the complete methodology and per-class breakdown.
