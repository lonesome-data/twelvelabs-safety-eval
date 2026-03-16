# Twelve Labs Marengo 3.0 and Pegasus 1.2 Evaluation on Worker Safety Video Classification

## Experiment Overview

This report documents an independent evaluation of Twelve Labs' two video foundation models — Marengo 3.0 (multimodal embeddings) and Pegasus 1.2 (video language model) — on industrial worker safety video classification. The evaluation was conducted March 10–11, 2026 by Sean Forester.

Two runs were performed. Run 1 (March 9–10) used a single dual-model index and achieved only 69% indexing success and 41% Pegasus labeling success due to API rate limits. Run 2 (March 10–11), documented here, used a split-index architecture and achieved **100% success** on all 40 videos across both models.

## Models Under Test

### Marengo 3.0

Multimodal embedding model producing 512-dimensional vectors from video, audio, text, or image inputs. Supports videos up to 4 hours with a minimum of 4 seconds. Used here for embedding quality assessment and text-to-video retrieval with visual and audio modalities enabled.

### Pegasus 1.2

Generative video language model. Takes an indexed video and a text prompt, produces natural language output. Used here as a zero-shot classifier with a constrained prompt at temperature 0.1.

## Dataset

### Source

Voxel51/Safe_and_Unsafe_Behaviours (HuggingFace, CC-BY-4.0). Originally published as:

> Önal, O., & Dandıl, E. (2024). Video dataset for the detection of safe and unsafe behaviours in workplaces. *Data in Brief*, 56, 110791. DOI: 10.1016/j.dib.2024.110791

The dataset contains 691 video clips from real surveillance cameras at Kafaoğlu Metal Plastik Makine San. ve Tic. A.Ş., a production facility in Eskişehir, Turkey. Videos were captured between November 5 and December 13, 2022 (39 days) from two UNV IP cameras at 1920×1080, 24fps, H.264/MP4. All employees provided informed consent; the study was approved by the Bilecik Şeyh Edebali University ethics committee.

### Class Taxonomy (8 classes)

| Safe Behaviors | Unsafe Behaviors |
|---|---|
| Safe Walkway — worker stays within designated path | Safe Walkway Violation — worker outside designated path |
| Authorized Intervention — properly equipped for maintenance | Unauthorized Intervention — no safety gear/authorization |
| Closed Panel Cover — panel properly closed | Opened Panel Cover — panel left open |
| Safe Carrying — forklift carrying ≤2 blocks | Carrying Overload with Forklift — forklift carrying >2 blocks |

### Dataset Limitations (from original paper)

- **Single facility**: One factory, two cameras, 39 days. Cannot generalize to other environments.
- **No inter-annotator agreement**: Labels assigned by factory experts without reported kappa or validation metrics.
- **Severe class imbalance**: Walkway Violation has 210 videos; Closed Panel Cover has 32 (6.6:1 ratio).
- **Ambiguous boundaries**: Authors acknowledge projection constraints make walkway violation labeling imprecise.
- **No baseline models**: The original paper is a dataset descriptor with zero benchmarks.

### Evaluation Subset

From the 125-video test split, videos shorter than 4 seconds were filtered (Twelve Labs minimum), leaving 116 eligible. Five videos per label were sampled using stratified selection sorted by duration ascending (shortest first) with seed=42, yielding **40 evaluation videos** with total duration of 257.9 seconds.

## Methodology

### Split-Index Architecture

To maximize successful API calls within the free tier's **50 requests/day** indexing limit, two separate single-model indexes were created:

- `safety-eval-marengo` — Marengo 3.0 only
- `safety-eval-pegasus` — Pegasus 1.2 only

This avoids the 2× duration multiplier from dual-model indexing and allows independent budget tracking per model. Each index consumed 257.9 seconds of its 3600-second daily duration budget.

### Pipeline

1. Download dataset from HuggingFace via FiftyOne
2. Filter test split for videos ≥4 seconds; stratified-sample 5 per label (sorted by duration ascending)
3. Index all 40 videos into Marengo index (40 of 50 daily requests)
4. Index all 40 videos into Pegasus index (30 requests on day 2 after quota reset)
5. Retrieve 512-dim Marengo embeddings via `indexes.videos.retrieve` with `embedding_option=["visual"]`
6. Classify each video via Pegasus `analyze` endpoint with constrained prompt at temperature=0.1
7. Run text-to-video search queries via `search.query` endpoint
8. Compute all metrics

### Rate Limiting Strategy

- 10-second minimum interval between all API calls (proactive throttle)
- Escalating backoff on 429 errors: 65s → 65s → 10 min (after 3 consecutive 429s)
- Daily quota detection: parses `x-ratelimit-request-remaining` header and bails immediately when exhausted
- State persistence to `eval_state.json` for resumable runs across quota resets

### Benchmark A: Embedding Quality (Marengo 3.0)

k-Nearest Neighbors classifier (k=5, cosine distance) evaluated with stratified k-fold cross-validation. Silhouette score computed for cluster cohesion. Binary version collapses 8 classes into safe vs. unsafe.

### Benchmark B: Retrieval Quality (Marengo 3.0)

Ten natural language queries — eight targeting specific classes, two cross-cutting (all-safe, all-unsafe). Top-5 results evaluated for Average Precision@5 and Recall@5.

### Benchmark C: Generation Quality (Pegasus 1.2)

Zero-shot classification with constrained prompt:

> "You are a worker-safety video classifier. Classify this video into exactly one label. UNSAFE: Safe Walkway Violation, Unauthorized Intervention, Opened Panel Cover, Carrying Overload with Forklift. SAFE: Safe Walkway, Authorized Intervention, Closed Panel Cover, Safe Carrying. Return ONLY the exact label name. No punctuation, no explanation, no extra words."

Metrics: 8-class accuracy, binary accuracy, format compliance, hallucination rate.

## Results

### Benchmark A: Embedding Quality

| Metric | Value | Interpretation |
|---|---|---|
| kNN 8-class accuracy (CV) | **0.425** | 3.4× random (0.125), but wrong more than right |
| kNN binary (safe/unsafe) | **0.600** | Marginally better than coin flip (0.50) |
| Silhouette score | **0.033** | ≈0 means classes completely overlap in vector space |

### Benchmark B: Retrieval Quality

| Metric | Value |
|---|---|
| mAP@5 | **0.374** |
| Recall@5 | **0.270** |

**Per-query breakdown:**

| Query | AP@5 | R@5 | Pattern |
|---|---|---|---|
| forklift safely transporting a normal load | **1.000** | **1.000** | Object retrieval — perfect |
| proper safety protocol being followed | 0.533 | 0.150 | Cross-cutting — moderate |
| authorized maintenance worker servicing equipment | 0.500 | 0.200 | Object retrieval — good |
| safety violation on the factory floor | 0.478 | 0.150 | Cross-cutting — moderate |
| worker stepping into a restricted area | 0.367 | 0.400 | State discrimination — weak |
| worker walking within the designated safe path | 0.333 | 0.200 | State discrimination — weak |
| forklift carrying too many items stacked high | 0.325 | 0.400 | Quantity discrimination — weak |
| electrical panel left open and unattended | 0.200 | 0.200 | State discrimination — poor |
| person walking outside the safe walkway boundary | **0.000** | **0.000** | State discrimination — fails |
| panel cover properly closed and secured | **0.000** | **0.000** | State discrimination — fails |

**Pattern**: Marengo excels at object-level retrieval ("find forklifts") but fails at state discrimination ("is the panel open or closed?", "is the worker inside or outside the walkway?").

### Benchmark C: Generation Quality

| Metric | Value | Interpretation |
|---|---|---|
| 8-class accuracy | **0.125** | Equivalent to random guessing |
| Binary accuracy | **0.150** | Worse than coin flip |
| Format compliance | **0.300** | 70% of outputs are invalid labels |
| Hallucination rate | **0.700** | 28 of 40 responses outside the valid label set |

**Pegasus output distribution (40 videos):**

| Output | Count | % | Valid Label? |
|---|---|---|---|
| "Unsafe" | 21 | 52.5% | No |
| "Safe Carrying" | 8 | 20.0% | Yes |
| "Safe Walkway" | 4 | 10.0% | Yes |
| "SAFE" | 2 | 5.0% | No |
| "Unsafe" (various) | 5 | 12.5% | No |

### Per-Class Pegasus Results

| Class | Ground Truth | Pegasus Prediction | Correct | Notes |
|---|---|---|---|---|
| Carrying Overload w/ Forklift (5) | Unsafe | Safe Carrying ×4, Safe Walkway ×1 | **0/5** | Misclassifies as safe — dangerous |
| Closed Panel Cover (5) | Safe | Unsafe ×4, Safe Walkway ×1 | **0/5** | Safe behavior flagged as unsafe |
| Opened Panel Cover (5) | Unsafe | Unsafe ×5 | **0/5** | Directionally right, wrong label |
| Authorized Intervention (5) | Safe | Unsafe ×5 | **0/5** | Safe behavior flagged as unsafe |
| Unauthorized Intervention (5) | Unsafe | Unsafe ×4, SAFE ×1 | **0/5** | Can't distinguish auth from unauth |
| Safe Walkway (5) | Safe | Unsafe ×4, Safe Walkway ×1 | **1/5** | 80% false alarm rate |
| Safe Walkway Violation (5) | Unsafe | Unsafe ×3, Safe Walkway ×1, SAFE ×1 | **0/5** | Violation called "Safe" — dangerous |
| Safe Carrying (5) | Safe | Safe Carrying ×4, Unsafe ×1 | **4/5** | Best class — visually distinctive |

## Analysis

### Marengo 3.0: Object Recognition Without State Understanding

Marengo's embeddings carry weak but nonzero class signal (kNN accuracy 3.4× random). However, the near-zero silhouette score (0.033) confirms that classes are not separable in embedding space. The retrieval results expose the fundamental limitation: Marengo recognizes *what* is in the scene (forklifts, panels, workers) but cannot determine *the state* of those objects (open vs. closed, inside vs. outside boundary, overloaded vs. normal).

For industrial safety, the difference between safe and unsafe is almost always about state, not presence. This makes Marengo unsuitable as the sole classification signal for safety monitoring, though it may serve as a useful pre-filter in a multi-stage pipeline.

### Pegasus 1.2: Binary Bias With Dangerous Misclassifications

The 70% hallucination rate is the headline finding. Despite explicit instructions to return one of eight exact labels, Pegasus defaults to "Unsafe" as a catch-all binary judgment 52.5% of the time. This is not one of the valid labels.

The most dangerous failure modes:
- **Carrying Overload → "Safe Carrying"** (4/5): Overloaded forklifts classified as safe
- **Safe Walkway Violation → "Safe Walkway" or "SAFE"** (2/5): Workers outside safe zones classified as safe
- **Authorized Intervention → "Unsafe"** (5/5): Properly authorized work flagged as violations

These are not edge cases — they represent systematic failures. In production, misclassifying violations as safe creates false confidence that is worse than having no monitoring system.

### Comparison: Run 1 vs Run 2

| Metric | Run 1 (partial, 50/30) | Run 2 (complete, 40/40) |
|---|---|---|
| Videos evaluated | 50 Marengo / 30 Pegasus | 40 Marengo / 40 Pegasus |
| API success rate | 58% | **100%** |
| kNN 8-class | 0.499 | 0.425 |
| kNN binary | 0.640 | 0.600 |
| Silhouette | 0.024 | 0.033 |
| mAP@5 | 0.426 | 0.374 |
| Pegasus 8-class | 0.200 | **0.125** |
| Pegasus hallucination | 0.600 | **0.700** |

Run 2's Pegasus metrics are *worse* than Run 1's partial results, suggesting the 10-video subset from Run 1 was not representative. The complete 40-video evaluation provides a more accurate (and more damning) picture.

### Confounds and Caveats

1. **Camera-class correlation**: Camera 9 captures walkway + panel classes; Camera 14 captures forklift classes. Marengo may partly be learning camera view, not behavior.
2. **Duration-class correlation**: Shortest-first sampling biases toward shorter classes (Closed Panel Cover avg 4.6s vs Unauthorized Intervention avg 10.2s).
3. **Single environment**: All results are specific to one Turkish factory with two camera angles.
4. **API non-determinism**: Pegasus at temperature=0.1 is not fully deterministic. Results may vary across runs.
5. **No negative controls**: We did not test with out-of-distribution video.
6. **Small sample size**: 5 videos per class limits statistical confidence. No confidence intervals computed.

## Conclusions

### Fitness for Production Safety Monitoring

**Marengo 3.0**: Not fit as sole classifier. Useful for object-level video retrieval and as a feature extractor in a multi-stage pipeline, but cannot distinguish safe from unsafe states of the same objects. Would need to be combined with specialized computer vision models for state detection.

**Pegasus 1.2**: Not fit for classification tasks. The 70% hallucination rate and systematic misclassification of violations as safe behavior represent unacceptable risk for safety-critical applications. The lack of structured output (constrained decoding, JSON mode) makes it unreliable for any classification workflow.

### What Would Be Needed

- Fine-tuning capability for domain-specific embedding spaces
- Structured output / constrained decoding for Pegasus classification
- On-premise deployment option for air-gapped industrial environments
- Published benchmarks on industrial/surveillance domains (not just sports)

## Reproducibility

### Requirements

Python 3.13, twelvelabs 1.2.0, numpy, scikit-learn, fiftyone, huggingface_hub ≥0.20.0, umap-learn.

### Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TWELVE_LABS_API_KEY=your_key

# Full evaluation (40 videos, requires 2 days for quota: 40+30+10 search+40 analyze = ~120 requests)
python eval_harness.py --split test --samples-per-label 5

# Quick sanity check (24 videos, fits in 1 day)
python eval_harness.py --split test --samples-per-label 3
```

### Notes

- First run downloads ~9.5 GB from HuggingFace
- State persists to `eval_state.json` — interrupted runs resume without re-indexing
- Random seed 42 for stratified sampling
- Free tier: 50 indexing requests/day, 50 analyze requests/day, 50 search requests/day
- Full eval requires 2 days due to 50 req/day indexing limit (40 Marengo + 40 Pegasus = 80 index requests)

## Per-Sample Results

### All 40 Pegasus Classifications

| Video | Ground Truth | Pegasus Output | Correct? |
|---|---|---|---|
| 3_te7.mp4 | Carrying Overload with Forklift | Safe Carrying | No |
| 3_te5.mp4 | Carrying Overload with Forklift | Safe Carrying | No |
| 3_te6.mp4 | Carrying Overload with Forklift | Safe Carrying | No |
| 3_te4.mp4 | Carrying Overload with Forklift | Safe Walkway | No |
| 3_te3.mp4 | Carrying Overload with Forklift | Safe Carrying | No |
| 6_te5.mp4 | Closed Panel Cover | Unsafe | No |
| 6_te4.mp4 | Closed Panel Cover | Unsafe | No |
| 6_te3.mp4 | Closed Panel Cover | Unsafe | No |
| 6_te2.mp4 | Closed Panel Cover | Safe Walkway | No |
| 6_te1.mp4 | Closed Panel Cover | Unsafe | No |
| 2_te2.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te3.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te4.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te5.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te6.mp4 | Opened Panel Cover | Unsafe | No |
| 5_te10.mp4 | Authorized Intervention | Unsafe | No |
| 5_te11.mp4 | Authorized Intervention | Unsafe | No |
| 5_te9.mp4 | Authorized Intervention | Unsafe | No |
| 5_te1.mp4 | Authorized Intervention | Unsafe | No |
| 5_te2.mp4 | Authorized Intervention | Unsafe | No |
| 1_te11.mp4 | Unauthorized Intervention | Unsafe | No |
| 1_te10.mp4 | Unauthorized Intervention | SAFE | No |
| 1_te9.mp4 | Unauthorized Intervention | Unsafe | No |
| 1_te8.mp4 | Unauthorized Intervention | Unsafe | No |
| 1_te7.mp4 | Unauthorized Intervention | Unsafe | No |
| 4_te2.mp4 | Safe Walkway | Unsafe | No |
| 4_te3.mp4 | Safe Walkway | Safe Walkway | Yes |
| 4_te14.mp4 | Safe Walkway | Unsafe | No |
| 4_te1.mp4 | Safe Walkway | Unsafe | No |
| 4_te12.mp4 | Safe Walkway | Unsafe | No |
| 0_te10.mp4 | Safe Walkway Violation | Unsafe | No |
| 0_te11.mp4 | Safe Walkway Violation | Unsafe | No |
| 0_te12.mp4 | Safe Walkway Violation | SAFE | No |
| 0_te13.mp4 | Safe Walkway Violation | Safe Walkway | No |
| 0_te14.mp4 | Safe Walkway Violation | Unsafe | No |
| 7_te4.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te3.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te2.mp4 | Safe Carrying | Unsafe | No |
| 7_te1.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te8.mp4 | Safe Carrying | Safe Carrying | Yes |

**Total: 5 correct / 40 = 12.5% accuracy**

## Citation

If referencing this evaluation:

> Forester, S. (2026). Independent Evaluation of Twelve Labs Marengo 3.0 and Pegasus 1.2 on Worker Safety Video Classification. Unpublished technical report.

Dataset citation:

> Önal, O., & Dandıl, E. (2024). Video dataset for the detection of safe and unsafe behaviours in workplaces. *Data in Brief*, 56, 110791. https://doi.org/10.1016/j.dib.2024.110791
