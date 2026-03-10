# Twelve Labs Marengo 3.0 and Pegasus 1.2 Evaluation on Worker Safety Video Classification

## Experiment Overview

This report documents an independent evaluation of Twelve Labs' two video foundation models, Marengo 3.0 and Pegasus 1.2, on industrial worker safety video classification. The goal was to assess whether these models are fit for production safety monitoring applications before considering a role at Twelve Labs.

The evaluation was conducted on March 9-10, 2026 by Sean Forester.

## Models Under Test

### Marengo 3.0

Marengo 3.0 is a multimodal embedding model. It takes video, audio, text, or image inputs and produces 512-dimensional vector embeddings suitable for similarity search, clustering, and downstream machine learning. It supports videos up to 4 hours in duration with a minimum of 4 seconds. It supports 37 languages. In this evaluation, both visual and audio modalities were enabled.

### Pegasus 1.2

Pegasus 1.2 is a video language model. It is a generative model that takes an indexed video and a text prompt, then produces a natural language response. It can describe, summarize, answer questions about, and classify video content. In this evaluation, it was used as a zero-shot classifier with a constrained prompt at temperature 0.1 to minimize randomness.

## Dataset

The dataset used is Voxel51/Safe_and_Unsafe_Behaviours, available on HuggingFace under the CC-BY-4.0 license. The original paper is published in ScienceDirect.

The dataset contains 691 video clips of simulated industrial workplace scenarios at 1920 by 1080 resolution, 24 frames per second, with durations ranging from 1 to 20 seconds. There are 566 training videos and 125 test videos.

The eight classes are divided into four safe and four unsafe behaviors:

Safe behaviors: Safe Walkway (worker stays within designated walkway), Authorized Intervention (worker properly equipped for equipment intervention), Closed Panel Cover (panel cover properly closed), and Safe Carrying (forklift carrying two or fewer blocks).

Unsafe behaviors: Safe Walkway Violation (worker exceeds safe walkway boundaries), Unauthorized Intervention (worker intervenes without safety gear or authorization), Opened Panel Cover (panel cover left open), and Carrying Overload with Forklift (forklift carrying three or more blocks).

### Evaluation Subset

From the test split, videos shorter than 4 seconds were filtered out due to Twelve Labs' minimum duration requirement, leaving 116 eligible videos. Ten videos per label were sampled using a fixed random seed of 42 for reproducibility, yielding 72 target videos.

Of these 72, only 50 were successfully indexed due to API rate limits on the free tier. Of the 50 indexed, only 30 received Pegasus labels for the same reason. The remaining 20 returned errors because the analyze endpoint was rate-limited at 8 requests per minute.

## Methodology

### Pipeline

Step one: download the dataset from HuggingFace via FiftyOne's load_from_hub function. Step two: filter the test split for videos at least 4 seconds in duration. Step three: stratified sample up to 10 videos per label with a fixed seed. Step four: create a Twelve Labs index configured with both Marengo 3.0 and Pegasus 1.2 using visual and audio modalities. Step five: upload and index each video using the tasks.create endpoint, then wait for completion with tasks.wait_for_done. Step six: retrieve 512-dimensional Marengo embeddings using the indexes.videos.retrieve endpoint with the visual embedding option. For videos with multiple segments, segment embeddings were averaged. Step seven: classify each video using the Pegasus analyze endpoint with a constrained prompt. Step eight: run text-to-video search queries using the search.query endpoint. Step nine: compute all metrics.

### Benchmark A: Embedding Quality

This benchmark tests whether Marengo embeddings can separate the eight safety classes in vector space. A k-Nearest Neighbors classifier with k equals 5 and cosine distance was evaluated using stratified k-fold cross-validation. The silhouette score was computed to measure cluster cohesion versus separation, where a score near 1 means well-separated clusters and a score near 0 means overlapping clusters. A binary version was also tested collapsing the eight classes into safe versus unsafe.

### Benchmark B: Retrieval Quality

This benchmark tests Marengo's text-to-video search capability. Ten natural language queries were crafted, eight targeting specific classes and two cross-cutting queries for general safety versus violation. For each query, the top 5 search results were checked against ground-truth labels to compute Average Precision at 5 and Recall at 5.

The ten queries were: "person walking outside the safe walkway boundary" expecting Safe Walkway Violation, "worker stepping into a restricted area without authorization" expecting Unauthorized Intervention, "electrical panel left open and unattended" expecting Opened Panel Cover, "forklift carrying too many items stacked dangerously high" expecting Carrying Overload with Forklift, "worker walking within the designated safe path" expecting Safe Walkway, "authorized maintenance worker servicing equipment" expecting Authorized Intervention, "panel cover properly closed and secured" expecting Closed Panel Cover, "forklift safely transporting a normal load" expecting Safe Carrying, "safety violation on the factory floor" expecting all unsafe labels, and "proper safety protocol being followed" expecting all safe labels.

### Benchmark C: Generation Quality

This benchmark tests Pegasus as a zero-shot video classifier. The exact prompt used was:

"You are a worker-safety video classifier. Classify this video into exactly one label. UNSAFE: Safe Walkway Violation, Unauthorized Intervention, Opened Panel Cover, Carrying Overload with Forklift. SAFE: Safe Walkway, Authorized Intervention, Closed Panel Cover, Safe Carrying. Return ONLY the exact label name. No punctuation, no explanation, no extra words."

Temperature was set to 0.1 for maximum determinism. Four metrics were computed: eight-class exact match accuracy, binary safe versus unsafe accuracy, format compliance (did it return a valid label from the eight options), and hallucination rate (percentage returning something outside the valid set).

## Results

### Benchmark A: Embedding Quality

The kNN eight-class accuracy was 0.499. For context, random guessing across eight classes would yield 0.125, so Marengo embeddings do carry some class signal, but less than 50 percent accuracy means the model is wrong more often than right.

The kNN binary accuracy collapsing to safe versus unsafe was 0.640. This is only slightly better than a coin flip of 0.50.

The silhouette score was 0.024. This is essentially zero, meaning the eight classes are almost completely overlapping in embedding space. There is no meaningful cluster structure.

### Benchmark B: Retrieval Quality

The mean Average Precision at 5 across all ten queries was 0.426. The mean Recall at 5 was 0.195.

Per-query results reveal a stark pattern. Queries about identifiable objects scored well: "forklift safely transporting a normal load" achieved a perfect AP of 1.00 and Recall of 0.83. "Authorized maintenance worker servicing equipment" scored AP 0.89 and Recall 0.40. "Electrical panel left open and unattended" scored AP 0.83 and Recall 0.20. "Proper safety protocol being followed" scored AP 0.89 and Recall 0.15.

Queries requiring state or boundary discrimination scored zero: "person walking outside the safe walkway boundary" returned AP 0.00 and Recall 0.00. "Worker stepping into a restricted area without authorization" returned AP 0.00 and Recall 0.00. "Worker walking within the designated safe path" returned AP 0.00 and Recall 0.00. "Panel cover properly closed and secured" returned AP 0.00 and Recall 0.00.

The remaining queries scored moderately: "forklift carrying too many items stacked dangerously high" returned AP 0.33 and Recall 0.29. "Safety violation on the factory floor" returned AP 0.33 and Recall 0.09.

The interpretation is clear: Marengo can recognize objects like forklifts, panels, and workers, but it cannot distinguish the state of those objects. It cannot tell open from closed, inside the walkway from outside the walkway, or authorized from unauthorized.

### Benchmark C: Generation Quality

Of 50 indexed videos, only 30 received Pegasus classifications. The other 20 returned errors due to rate limiting.

The eight-class accuracy was 0.200, meaning Pegasus correctly identified the exact label for only 6 of 30 videos. The binary safe versus unsafe accuracy was 0.300. Format compliance was 0.400, meaning only 12 of 30 responses were valid labels from the eight-class taxonomy. The hallucination rate was 0.600.

The distribution of Pegasus outputs across all 30 non-error responses was as follows. It returned "Unsafe" 17 times, which is 57 percent of all responses. "Unsafe" is not one of the eight valid labels. It returned "Safe Walkway" 8 times, "Safe Carrying" 4 times, and "SAFE" once with wrong casing.

### Per-Class Pegasus Results

Authorized Intervention: All 10 videos failed due to rate limits. Zero data.

Carrying Overload with Forklift: All 7 videos failed due to rate limits. Zero data.

Closed Panel Cover: 5 of 8 received labels. Zero correct. Pegasus predicted Safe Walkway 3 times and Unsafe 2 times. This is a safe behavior being called either wrong or unsafe.

Opened Panel Cover: 10 of 10 received labels. Zero correct. Pegasus predicted Unsafe 9 times and Safe Walkway once. Directionally, calling an open panel "Unsafe" is not wrong, but it does not match the specific label and reveals that Pegasus is defaulting to a binary judgment rather than doing eight-class classification.

Safe Carrying: 6 of 7 received labels. Four correct. Pegasus predicted Safe Carrying 4 times and Unsafe twice. This is the best-performing class, likely because a forklift carrying a small load is visually distinctive.

Safe Walkway: 3 of 10 received labels. Two correct. Pegasus predicted Safe Walkway twice and Unsafe once.

Safe Walkway Violation: 3 of 10 received labels. Zero correct. Pegasus predicted Safe Walkway twice and Unsafe once. This is the most dangerous failure mode: a safety violation being classified as safe behavior. In a production deployment, this means the system would fail to alert when a worker walks outside the designated safe area.

Unauthorized Intervention: 3 of 10 received labels. Zero correct. Pegasus predicted Unsafe twice and SAFE once. It cannot distinguish authorized from unauthorized work.

## Analysis

### Marengo 3.0 Strengths

Object-level retrieval works well. When you search for a forklift, a panel, or a worker near equipment, Marengo finds relevant videos. The 512-dimensional embeddings are compact and practical for downstream machine learning pipelines. The model supports videos up to 4 hours, 37 languages, and multimodal inputs including composed text-plus-image queries. Multi-segment temporal encoding provides video-level representations rather than just frame-level analysis.

### Marengo 3.0 Weaknesses

State discrimination is the fundamental weakness. In a safety context, the difference between safe and unsafe is almost always about the state of an object or the position of a person relative to a boundary, not the mere presence of that object or person. Marengo cannot distinguish open from closed panels, inside from outside walkways, or authorized from unauthorized workers. All state-based retrieval queries returned AP of 0.00.

The silhouette score of 0.024 confirms that the eight safety classes occupy nearly identical regions in the embedding space. This means any downstream classifier built on Marengo embeddings will struggle.

Twelve Labs publishes benchmarks heavily weighted toward sports domains. SoccerNet action recognition and basketball analytics show strong results. However, no industrial safety or surveillance benchmarks have been published, and our results suggest those domains perform significantly worse.

### Pegasus 1.2 Strengths

It can identify visually distinctive safe behaviors, particularly Safe Carrying where the forklift carries a small number of blocks. The per-call latency is reasonable once videos are indexed. The embed-then-query architecture means re-querying the same video is cheap.

### Pegasus 1.2 Weaknesses

The 60 percent hallucination rate is the headline finding. Despite being told to return only an exact label name from a list of eight options, Pegasus returned "Unsafe" 57 percent of the time, which is not one of the valid labels. It is treating the task as binary classification rather than eight-class classification. At temperature 0.1, which should produce near-deterministic outputs, this suggests the model genuinely cannot distinguish between the eight categories.

There is no structured output or constrained decoding mode. Unlike modern LLM APIs that offer JSON mode or tool-use to guarantee output format, Pegasus returns freeform text. For classification tasks, this is a significant limitation.

The most dangerous failure is "Safe Walkway Violation" being classified as "Safe Walkway" in two out of three cases. In production, this means the system would not alert when a worker leaves the designated safe area.

### Production Concerns

The free tier rate limit of 8 requests per minute across all endpoints made batch evaluation impractical. Of our 72 target videos, 22 failed to index and 20 failed Pegasus labeling. This is itself a product finding: batch workflows require a paid tier.

Twelve Labs is API-only with no on-premise or edge deployment option. For industrial safety monitoring in factories with limited connectivity or air-gapped environments, this is a blocker.

No fine-tuning is available for either model. Users cannot adapt the embedding space or the generative model to their specific safety categories.

The walkway violation misclassification, where unsafe behavior is labeled as safe, represents a liability risk in any real deployment. Safety monitoring systems that miss violations are worse than no system at all because they create false confidence.

## Reproducibility

### Requirements

Python 3.13, twelvelabs SDK version 1.2.0, numpy, scikit-learn, fiftyone, huggingface_hub version 0.20.0 or later, and umap-learn.

### Commands

Create and activate a virtual environment: python3 -m venv .venv, then source .venv/bin/activate. Install dependencies: pip install -r requirements.txt. Set the API key: export TWELVE_LABS_API_KEY with your key. Run a quick sanity check with 24 videos: python eval_harness.py --split test --samples-per-label 3. Run the full evaluation with 72 or more videos: python eval_harness.py --split test --samples-per-label 10.

### Notes

The first run downloads approximately 9.5 gigabytes of video data from HuggingFace. State is persisted to eval_state.json so interrupted runs resume without re-indexing or re-uploading. The random seed is 42 for stratified sampling. Results may vary slightly due to Pegasus non-determinism even at low temperature. Free tier rate limits cause approximately 20 percent failure rate on large batches.

## Raw Sample Results

The following table shows every video that received a Pegasus classification, along with its ground truth label and the model prediction.

### Closed Panel Cover (safe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 6_te13.mp4 | Closed Panel Cover | Unsafe | No |
| 6_te2.mp4 | Closed Panel Cover | Safe Walkway | No |
| 6_te4.mp4 | Closed Panel Cover | Unsafe | No |
| 6_te11.mp4 | Closed Panel Cover | Safe Walkway | No |
| 6_te12.mp4 | Closed Panel Cover | Safe Walkway | No |

### Opened Panel Cover (unsafe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 2_te3.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te11.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te8.mp4 | Opened Panel Cover | Safe Walkway | No |
| 2_te12.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te5.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te6.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te2.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te9.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te7.mp4 | Opened Panel Cover | Unsafe | No |
| 2_te10.mp4 | Opened Panel Cover | Unsafe | No |

### Safe Carrying (safe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 7_te4.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te6.mp4 | Safe Carrying | Unsafe | No |
| 7_te8.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te3.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te7.mp4 | Safe Carrying | Safe Carrying | Yes |
| 7_te2.mp4 | Safe Carrying | Unsafe | No |

### Safe Walkway (safe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 4_te3.mp4 | Safe Walkway | Safe Walkway | Yes |
| 4_te5.mp4 | Safe Walkway | Unsafe | No |
| 4_te18.mp4 | Safe Walkway | Safe Walkway | Yes |

### Safe Walkway Violation (unsafe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 0_te27.mp4 | Safe Walkway Violation | Safe Walkway | No |
| 0_te3.mp4 | Safe Walkway Violation | Safe Walkway | No |
| 0_te5.mp4 | Safe Walkway Violation | Unsafe | No |

### Unauthorized Intervention (unsafe behavior)

| Video | Ground Truth | Pegasus Prediction | Correct |
|---|---|---|---|
| 1_te11.mp4 | Unauthorized Intervention | Unsafe | No |
| 1_te10.mp4 | Unauthorized Intervention | SAFE | No |
| 1_te5.mp4 | Unauthorized Intervention | Unsafe | No |

### Authorized Intervention and Carrying Overload with Forklift

All 10 Authorized Intervention videos and all 7 Carrying Overload with Forklift videos failed to receive Pegasus classifications due to API rate limit exhaustion. These two classes have zero evaluation data from the generative model.
