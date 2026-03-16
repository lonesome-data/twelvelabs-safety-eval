#!/usr/bin/env python3
"""
Rigorous evaluation of TwelveLabs Marengo 3.0 embeddings and Pegasus 1.2 labeling.

What this script does that main.py doesn't:
1. Calls Pegasus on EVERY video (not just one per cluster) to get per-video predictions
2. Compares predictions against ground truth labels
3. Reports accuracy, per-class precision/recall/F1, confusion matrix
4. Evaluates embedding quality via clustering metrics (ARI, NMI, silhouette)
5. Tests whether Pegasus is actually understanding video or just guessing "Safe Walkway"
6. Runs KMeans with elbow method to check if k=8 is even appropriate

Usage:
    python evaluate.py                  # full evaluation (calls Pegasus per video)
    python evaluate.py --embeddings-only  # skip Pegasus, just evaluate embeddings
"""

import os
import json
import time
import argparse
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import fiftyone as fo
from fiftyone import ViewField as F

from twelvelabs import TwelveLabs
from twelvelabs.errors import TooManyRequestsError


# =============================================================================
# Config
# =============================================================================

load_dotenv()

VALID_LABELS = [
    "Safe Walkway Violation",
    "Unauthorized Intervention",
    "Opened Panel Cover",
    "Carrying Overload with Forklift",
    "Safe Walkway",
    "Authorized Intervention",
    "Closed Panel Cover",
    "Safe Carrying",
]

CLUSTER_LABEL_PROMPT = """
Analyze this workplace safety video and classify it as exactly ONE of the following labels.

UNSAFE BEHAVIORS (violations):
- Safe Walkway Violation: Worker is NOT using the designated safe walkway, walking in restricted/hazardous areas
- Unauthorized Intervention: Worker accessing or operating equipment without proper authorization
- Opened Panel Cover: Electrical/machinery panel left open unsafely
- Carrying Overload with Forklift: Forklift carrying excessive or unstable load

SAFE BEHAVIORS (good practices):
- Safe Walkway: Worker IS correctly using the designated safe walkway/pedestrian path
- Authorized Intervention: Worker properly authorized and following procedures for equipment access
- Closed Panel Cover: Electrical/machinery panels properly secured and closed
- Safe Carrying: Forklift operating with appropriate, stable load

Return ONLY the exact label name, nothing else.
"""

# Open-ended prompt — no answer key provided
OPEN_ENDED_PROMPT = """
Describe what is happening in this workplace safety video in one sentence.
Focus on: what the worker is doing, what equipment is involved, and whether
the behavior appears safe or unsafe.
"""


def load_dataset_and_client():
    dataset = fo.load_dataset("safe_unsafe_behaviours")
    client = TwelveLabs(api_key=os.getenv("TL_API_KEY"))
    indexed_view = dataset.exists("tl_video_id")
    print(f"Loaded {len(indexed_view)} indexed videos")
    return dataset, client, indexed_view


# =============================================================================
# 1. Per-video Pegasus labeling (constrained prompt)
# =============================================================================

def evaluate_pegasus_constrained(client, indexed_view):
    print("\n" + "=" * 70)
    print("PEGASUS EVALUATION: Constrained classification (answer key in prompt)")
    print("=" * 70)

    ground_truths = []
    predictions = []
    raw_responses = []

    for i, sample in enumerate(indexed_view.iter_samples(progress=True)):
        gt = sample.ground_truth.label if sample.ground_truth else "Unknown"
        ground_truths.append(gt)

        # Rate limit: 8 req/min, so wait 8s between calls
        if i > 0:
            time.sleep(8)

        raw = call_pegasus(client, sample.tl_video_id, CLUSTER_LABEL_PROMPT)
        raw_responses.append(raw)

        pred = normalize_label(raw)
        predictions.append(pred)

        match = "✓" if pred == gt else "✗"
        print(f"  {match} GT: {gt:<35} Pred: {pred:<35} Raw: {raw[:60]}")

    print_classification_metrics(ground_truths, predictions, "Constrained Pegasus")
    return ground_truths, predictions, raw_responses


def evaluate_pegasus_open_ended(client, indexed_view):
    print("\n" + "=" * 70)
    print("PEGASUS EVALUATION: Open-ended (no answer key)")
    print("=" * 70)

    for i, sample in enumerate(indexed_view.iter_samples(progress=True)):
        gt = sample.ground_truth.label if sample.ground_truth else "Unknown"

        if i > 0:
            time.sleep(8)

        raw = call_pegasus(client, sample.tl_video_id, OPEN_ENDED_PROMPT)
        print(f"  GT: {gt:<35} Description: {raw[:120]}")


def call_pegasus(client, video_id, prompt, temperature=0.2, max_retries=5):
    for attempt in range(max_retries):
        try:
            result = client.analyze(video_id=video_id, prompt=prompt, temperature=temperature)
            return result.data.strip()
        except TooManyRequestsError as e:
            retry_after = int(e.headers.get("retry-after", 15)) + 2
            print(f"    Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after)
    raise RuntimeError(f"Failed after {max_retries} retries")


def normalize_label(raw):
    raw_lower = raw.lower().strip()
    for label in VALID_LABELS:
        if label.lower() == raw_lower:
            return label
    # Fuzzy: check if any valid label is contained in the response
    for label in VALID_LABELS:
        if label.lower() in raw_lower:
            return label
    return raw


# =============================================================================
# 2. Embedding quality evaluation
# =============================================================================

def evaluate_embeddings(indexed_view):
    print("\n" + "=" * 70)
    print("EMBEDDING EVALUATION: Marengo 3.0 (512-dim)")
    print("=" * 70)

    embeddings = np.array(indexed_view.values("tl_embedding"))
    ground_truths = []
    for sample in indexed_view.iter_samples():
        ground_truths.append(sample.ground_truth.label if sample.ground_truth else "Unknown")

    n_samples = len(embeddings)
    n_classes = len(set(ground_truths))
    print(f"\nSamples: {n_samples}, Classes: {n_classes}, Dims: {embeddings.shape[1]}")

    # Basic embedding stats
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms: mean={norms.mean():.3f}, std={norms.std():.3f}, "
          f"min={norms.min():.3f}, max={norms.max():.3f}")

    # Pairwise cosine similarities
    normed = embeddings / norms[:, np.newaxis]
    cos_sim = normed @ normed.T

    # Within-class vs between-class similarity
    gt_arr = np.array(ground_truths)
    within_sims = []
    between_sims = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if gt_arr[i] == gt_arr[j]:
                within_sims.append(cos_sim[i, j])
            else:
                between_sims.append(cos_sim[i, j])

    print(f"\nWithin-class cosine similarity:  mean={np.mean(within_sims):.4f}, "
          f"std={np.std(within_sims):.4f} (n={len(within_sims)} pairs)")
    print(f"Between-class cosine similarity: mean={np.mean(between_sims):.4f}, "
          f"std={np.std(between_sims):.4f} (n={len(between_sims)} pairs)")

    separation = np.mean(within_sims) - np.mean(between_sims)
    print(f"Separation (within - between):   {separation:.4f}")
    if separation <= 0:
        print("  ⚠ NEGATIVE SEPARATION: embeddings do NOT cluster by class")
    elif separation < 0.05:
        print("  ⚠ WEAK SEPARATION: embeddings barely distinguish classes")
    else:
        print(f"  ✓ Positive separation ({separation:.4f})")

    # KMeans clustering with ground truth comparison
    print(f"\n--- KMeans Clustering (k={n_classes}) ---")
    if n_samples > n_classes:
        kmeans = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        ari = adjusted_rand_score(ground_truths, cluster_labels)
        nmi = normalized_mutual_info_score(ground_truths, cluster_labels)
        print(f"Adjusted Rand Index (ARI):         {ari:.4f}  (1.0=perfect, 0.0=random)")
        print(f"Normalized Mutual Information:      {nmi:.4f}  (1.0=perfect, 0.0=random)")

        if n_samples > n_classes + 1:
            sil = silhouette_score(embeddings, cluster_labels)
            ch = calinski_harabasz_score(embeddings, cluster_labels)
            db = davies_bouldin_score(embeddings, cluster_labels)
            print(f"Silhouette Score:                   {sil:.4f}  (1.0=dense+separated, 0=overlapping)")
            print(f"Calinski-Harabasz Index:            {ch:.1f}   (higher=better)")
            print(f"Davies-Bouldin Index:               {db:.4f}  (lower=better, 0=perfect)")

    # Elbow method
    print(f"\n--- Elbow Method (is k=8 even right?) ---")
    max_k = min(n_samples - 1, 12)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(embeddings)
        print(f"  k={k:2d}  inertia={km.inertia_:.1f}")

    return embeddings, ground_truths


# =============================================================================
# 3. Baseline comparisons
# =============================================================================

def print_baselines(ground_truths):
    print("\n" + "=" * 70)
    print("BASELINES: What does random / majority-class performance look like?")
    print("=" * 70)

    n = len(ground_truths)
    n_classes = len(set(ground_truths))
    counts = Counter(ground_truths)
    majority_class = counts.most_common(1)[0][0]
    majority_count = counts.most_common(1)[0][1]

    print(f"Random baseline accuracy:    {1/n_classes:.1%} (1/{n_classes})")
    print(f"Majority-class baseline:     {majority_count/n:.1%} (always predict '{majority_class}')")
    print(f"\nClass distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<35} {count:3d} ({count/n:.1%})")


# =============================================================================
# Metrics
# =============================================================================

def print_classification_metrics(ground_truths, predictions, name):
    print(f"\n--- {name}: Classification Metrics ---")

    acc = accuracy_score(ground_truths, predictions)
    print(f"Overall Accuracy: {acc:.1%} ({sum(1 for g,p in zip(ground_truths, predictions) if g==p)}/{len(ground_truths)})")

    pred_distribution = Counter(predictions)
    print(f"\nPrediction distribution:")
    for label, count in sorted(pred_distribution.items()):
        print(f"  {label:<35} {count:3d}")

    all_labels = sorted(set(ground_truths) | set(predictions))
    print(f"\n{classification_report(ground_truths, predictions, labels=all_labels, zero_division=0)}")

    print("Confusion Matrix (rows=ground truth, cols=predicted):")
    cm = confusion_matrix(ground_truths, predictions, labels=all_labels)
    # Print header
    short_labels = [l[:20] for l in all_labels]
    header = f"{'':>22}" + "".join(f"{s:>22}" for s in short_labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{short_labels[i]:>22}" + "".join(f"{v:>22}" for v in row))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate TwelveLabs models rigorously")
    parser.add_argument("--embeddings-only", action="store_true",
                        help="Skip Pegasus calls, only evaluate embeddings")
    args = parser.parse_args()

    dataset, client, indexed_view = load_dataset_and_client()

    # Always run embedding evaluation
    embeddings, ground_truths = evaluate_embeddings(indexed_view)
    print_baselines(ground_truths)

    if not args.embeddings_only:
        # Per-video Pegasus with constrained prompt
        gt, preds, raw = evaluate_pegasus_constrained(client, indexed_view)

        # Open-ended Pegasus (no answer key) — qualitative check
        evaluate_pegasus_open_ended(client, indexed_view)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
