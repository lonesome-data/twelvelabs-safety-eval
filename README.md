# Twelve Labs Video Safety Evaluation

Independent benchmark of **Marengo 3.0** (embeddings/retrieval) and **Pegasus 1.2** (generation) on industrial worker safety video classification.

**Verdict: Neither model is production-ready for automated safety monitoring.** Marengo embeddings capture object-level semantics but fail at state discrimination. Pegasus hallucinates 70% of labels despite constrained prompting.

## Results

| Benchmark | Metric | Score |
|---|---|---|
| Marengo Embeddings | kNN 8-class accuracy | **0.425** |
| Marengo Embeddings | kNN binary (safe/unsafe) | **0.600** |
| Marengo Embeddings | Silhouette score | **0.033** |
| Marengo Retrieval | mAP@5 | **0.374** |
| Marengo Retrieval | Recall@5 | **0.270** |
| Pegasus Generation | 8-class accuracy | **0.125** |
| Pegasus Generation | Format compliance | **0.300** |
| Pegasus Generation | Hallucination rate | **0.700** |

### What works

- **Object-level retrieval** — "forklift safely transporting a normal load" returns AP@5 = 1.0
- **Authorized worker queries** — AP@5 = 0.5 for maintenance worker identification
- 512-dim embeddings are compact and practical for downstream ML pipelines

### What fails

- **State discrimination** — cannot distinguish open vs closed panels, inside vs outside walkways (AP = 0.0)
- **Pegasus defaults to "Unsafe"** — hallucinated non-label response in 70% of cases
- **Critical safety misclassification** — unsafe events labeled safe, defeating the purpose of monitoring
- **Generation accuracy at chance level** — 12.5% on 8 classes (random baseline = 12.5%)

## Dataset

[Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) (CC-BY-4.0)
— Önal & Dandıl 2024, *Data in Brief*, [DOI 10.1016/j.dib.2024.110791](https://doi.org/10.1016/j.dib.2024.110791)

- 691 videos, 8 classes (4 safe / 4 unsafe), single Turkish factory
- 1920×1080, 24 fps, 1–20 seconds
- Evaluation subset: 40 videos (5 per class) from test split

| Safe | Unsafe |
|---|---|
| Safe Walkway | Safe Walkway Violation |
| Authorized Intervention | Unauthorized Intervention |
| Closed Panel Cover | Opened Panel Cover |
| Safe Carrying | Carrying Overload with Forklift |

## Methodology

### A. Embedding Quality (Marengo 3.0)

Evaluates whether 512-dim embeddings separate safety classes in vector space.

- **kNN classification** (k=5, cosine distance, StratifiedKFold cross-validation)
- **Silhouette score** for cluster cohesion — 0.033 indicates near-random overlap

### B. Retrieval Quality (Marengo 3.0)

Text-to-video search with 10 natural language queries mapped to ground-truth labels.

```
"forklift safely transporting a normal load"           → AP = 1.00  ✓
"authorized maintenance worker servicing equipment"    → AP = 0.50
"person walking outside the safe walkway boundary"     → AP = 0.00  ✗
"panel cover properly closed and secured"              → AP = 0.00  ✗
```

### C. Generation Quality (Pegasus 1.2)

Zero-shot classification with constrained prompt at temperature = 0.1.

Pegasus output distribution (40 responses):
```
"Unsafe"              — not a valid label (hallucinated)
"Safe Carrying"       — only correct category with recall
"Safe Walkway"        — often applied to violations
```

## Project Structure

```
├── eval_harness.py                      # Evaluation pipeline
├── eval_results.json                    # Per-sample results and aggregate metrics
├── experiment_report.json               # Structured report with methodology
├── full_experiment_report.md            # Narrative report
├── explore_results.ipynb                # Interactive analysis notebook
├── requirements.txt                     # Python dependencies
└── worker-safety-kit/
    ├── evaluation-report.tex            # LaTeX write-up
    ├── references.bib                   # Bibliography
    ├── evaluate.py                      # Standalone evaluation script
    └── main.py                          # Kit entrypoint
```

## Quickstart

```bash
git clone https://github.com/lonesome-data/twelvelabs-safety-eval.git
cd twelvelabs-safety-eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TWELVE_LABS_API_KEY=<your_key>

# Quick run (40 videos, ~10 min with rate limit pauses)
python eval_harness.py --split test --samples-per-label 5
```

First run downloads ~9.5 GB from HuggingFace. State persists to `eval_state.json` so interrupted runs resume without re-indexing.

## Rate Limits

Free tier: 50 requests/day per endpoint, 3600s total video duration/day. The harness implements proactive throttling and retry-after parsing. Batch evaluation workflows are impractical at this tier — this is itself a finding.

## Reproducibility

- Python 3.13, `twelvelabs` SDK 1.2.0
- Random seed: 42
- Split-index strategy: separate Marengo and Pegasus indexes
- Results may vary slightly due to Pegasus non-determinism at temperature = 0.1

## License

MIT
