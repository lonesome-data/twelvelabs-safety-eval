"""
Twelve Labs API Evaluation Harness
===================================
Tests Marengo 3.0 (embeddings) and Pegasus 1.2 (generation) on the
Voxel51/Safe_and_Unsafe_Behaviours dataset (691 videos, 8 classes).

Uses two separate indexes (Marengo-only and Pegasus-only) to maximize
video capacity within the free-tier index usage limits:
  - 13 min total indexed duration per 10-hour window
  - 6 min max indexed duration per index per 10-hour window
  - 50 videos max per index

Usage:
    1. export TWELVE_LABS_API_KEY=your_key
    2. pip install -r requirements.txt
    3. python eval_harness.py [--samples-per-label N] [--split train|test|both]

The dataset is downloaded automatically from HuggingFace via FiftyOne.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    classification_report,
    silhouette_score,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from twelvelabs import TwelveLabs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
DATASET_NAME = "safe_unsafe_behaviours"
HF_SLUG = "Voxel51/Safe_and_Unsafe_Behaviours"

LABELS = [
    "Safe Walkway",
    "Safe Walkway Violation",
    "Authorized Intervention",
    "Unauthorized Intervention",
    "Closed Panel Cover",
    "Opened Panel Cover",
    "Safe Carrying",
    "Carrying Overload with Forklift",
]

SAFE_LABELS = {"Safe Walkway", "Authorized Intervention", "Closed Panel Cover", "Safe Carrying"}
UNSAFE_LABELS = set(LABELS) - SAFE_LABELS

# Text queries mapped to expected ground-truth labels for retrieval eval.
RETRIEVAL_QUERIES: dict[str, list[str]] = {
    # Unsafe queries
    "person walking outside the safe walkway boundary":
        ["Safe Walkway Violation"],
    "worker stepping into a restricted area without authorization":
        ["Unauthorized Intervention"],
    "electrical panel left open and unattended":
        ["Opened Panel Cover"],
    "forklift carrying too many items stacked dangerously high":
        ["Carrying Overload with Forklift"],
    # Safe queries
    "worker walking within the designated safe path":
        ["Safe Walkway"],
    "authorized maintenance worker servicing equipment":
        ["Authorized Intervention"],
    "panel cover properly closed and secured":
        ["Closed Panel Cover"],
    "forklift safely transporting a normal load":
        ["Safe Carrying"],
    # Harder / cross-cutting queries
    "safety violation on the factory floor":
        list(UNSAFE_LABELS),
    "proper safety protocol being followed":
        list(SAFE_LABELS),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ["TWELVE_LABS_API_KEY"]
MARENGO_INDEX_NAME = "safety-eval-marengo"
PEGASUS_INDEX_NAME = "safety-eval-pegasus"
MIN_DURATION = 4.0
STATE_FILE = Path("eval_state.json")

# Duration budget per index (seconds). The free tier allows 6 min per index
# per 10-hour window. We use 5.5 min as a safety margin.
MAX_DURATION_PER_INDEX = 5.5 * 60  # 330 seconds

# Max videos per index (free tier allows 100, we cap at 50 to stay safe)
MAX_VIDEOS_PER_INDEX = 50


@dataclass
class VideoSample:
    filepath: str
    label: str
    sample_id: str
    duration: float = 0.0
    marengo_video_id: str | None = None
    pegasus_video_id: str | None = None
    embedding: list[float] | None = None
    pegasus_label: str | None = None


@dataclass
class BenchmarkResults:
    # A – Embedding
    embedding_accuracy_knn: float = 0.0
    embedding_silhouette: float = 0.0
    embedding_binary_accuracy: float = 0.0
    # B – Retrieval
    retrieval_map_at_5: float = 0.0
    retrieval_recall_at_5: float = 0.0
    retrieval_per_query: dict = field(default_factory=dict)
    # C – Generation
    generation_accuracy: float = 0.0
    generation_binary_accuracy: float = 0.0
    generation_format_compliance: float = 0.0
    generation_hallucination_rate: float = 0.0
    per_class_report: str = ""
    confusion: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        rq = "\n".join(
            f"      {q[:55]:55s}  AP={v['ap']:.2f}  R@5={v['recall']:.2f}"
            for q, v in self.retrieval_per_query.items()
        )
        return f"""
============================================================
                   BENCHMARK RESULTS
============================================================

A. EMBEDDING QUALITY (Marengo 3.0)
   kNN accuracy (CV):         {self.embedding_accuracy_knn:.3f}
   kNN binary (safe/unsafe):  {self.embedding_binary_accuracy:.3f}
   Silhouette score:          {self.embedding_silhouette:.3f}

B. RETRIEVAL QUALITY (Marengo 3.0)
   mAP@5:                     {self.retrieval_map_at_5:.3f}
   Recall@5:                  {self.retrieval_recall_at_5:.3f}
   Per-query breakdown:
{rq}

C. GENERATION QUALITY (Pegasus 1.2)
   8-class accuracy:          {self.generation_accuracy:.3f}
   Binary accuracy:           {self.generation_binary_accuracy:.3f}
   Format compliance:         {self.generation_format_compliance:.3f}
   Hallucination rate:        {self.generation_hallucination_rate:.3f}

D. PER-CLASS BREAKDOWN
{self.per_class_report}
============================================================
"""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_video_duration(filepath: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def load_fiftyone_dataset(split: str, samples_per_label: int) -> list[VideoSample]:
    """Download from HuggingFace and return a stratified sample, sorted by duration."""
    if fo.dataset_exists(DATASET_NAME):
        dataset = fo.load_dataset(DATASET_NAME)
        log.info(f"Loaded existing FiftyOne dataset '{DATASET_NAME}' ({len(dataset)} samples)")
    else:
        log.info(f"Downloading {HF_SLUG} from HuggingFace (~9.5 GB)...")
        dataset = load_from_hub(
            HF_SLUG,
            name=DATASET_NAME,
            persistent=True,
            overwrite=False,
        )
        log.info(f"Downloaded {len(dataset)} samples")

    # Filter by split tag and minimum duration
    if split == "both":
        view = dataset.match(F("metadata.duration") >= MIN_DURATION)
    else:
        view = (
            dataset
            .match_tags(split)
            .match(F("metadata.duration") >= MIN_DURATION)
        )

    log.info(f"After filtering (split={split}, dur>={MIN_DURATION}s): {len(view)} videos")

    # Stratified sampling — sort by duration within each label to prefer shorter videos
    samples = []
    for label in sorted(view.distinct("ground_truth.label")):
        label_view = view.match(F("ground_truth.label") == label)
        # Sort by duration ascending to pick shortest videos first
        label_view = label_view.sort_by("metadata.duration")
        n = min(samples_per_label, len(label_view))
        count = 0
        for s in label_view:
            if count >= n:
                break
            dur = s.metadata.duration if s.metadata and s.metadata.duration else 0.0
            samples.append(VideoSample(
                filepath=s.filepath,
                label=s.ground_truth.label,
                sample_id=str(s.id),
                duration=dur,
            ))
            count += 1
        log.info(f"  {label}: {count} videos sampled (of {len(label_view)} available)")

    # Sort all samples by duration ascending for optimal budget usage
    samples.sort(key=lambda s: s.duration)

    total_dur = sum(s.duration for s in samples)
    log.info(f"Total evaluation set: {len(samples)} videos, {total_dur:.1f}s total duration")
    log.info(f"Duration budget per index: {MAX_DURATION_PER_INDEX:.0f}s")

    return samples


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(samples: list[VideoSample]):
    data = [
        {
            "filepath": s.filepath,
            "label": s.label,
            "sample_id": s.sample_id,
            "duration": s.duration,
            "marengo_video_id": s.marengo_video_id,
            "pegasus_video_id": s.pegasus_video_id,
            "pegasus_label": s.pegasus_label,
        }
        for s in samples
    ]
    STATE_FILE.write_text(json.dumps(data, indent=2))


def restore_state(samples: list[VideoSample]) -> list[VideoSample]:
    if not STATE_FILE.exists():
        return samples
    saved = {d["filepath"]: d for d in json.loads(STATE_FILE.read_text())}
    restored = 0
    for s in samples:
        if s.filepath in saved:
            prev = saved[s.filepath]
            if prev.get("marengo_video_id"):
                s.marengo_video_id = prev["marengo_video_id"]
                restored += 1
            if prev.get("pegasus_video_id"):
                s.pegasus_video_id = prev["pegasus_video_id"]
            if prev.get("pegasus_label"):
                s.pegasus_label = prev["pegasus_label"]
            if prev.get("duration"):
                s.duration = prev["duration"]
    if restored:
        log.info(f"Restored state for {restored} previously-indexed videos")
    return samples


# ---------------------------------------------------------------------------
# Twelve Labs helpers
# ---------------------------------------------------------------------------

def get_or_create_index(client: TwelveLabs, index_name: str, model_name: str) -> str:
    for idx in client.indexes.list():
        if idx.index_name == index_name:
            log.info(f"Reusing existing index '{index_name}' ({idx.id})")
            return idx.id

    from twelvelabs.indexes.types import IndexesCreateRequestModelsItem

    idx = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name=model_name,
                model_options=["visual", "audio"],
            ),
        ],
    )
    log.info(f"Created index '{index_name}' ({idx.id}) with model {model_name}")
    return idx.id


def index_video(client: TwelveLabs, index_id: str, sample: VideoSample) -> str:
    with open(sample.filepath, "rb") as f:
        task = _rate_limited_call(
            client.tasks.create,
            _interval=10.0,
            _max_retries=8,
            index_id=index_id,
            video_file=f.read(),
            user_metadata=json.dumps({
                "label": sample.label,
                "sample_id": sample.sample_id,
            }),
        )
    task = client.tasks.wait_for_done(task_id=task.id)
    if task.status != "ready":
        raise RuntimeError(f"Indexing failed for {sample.filepath}: {task.status}")
    video_id = client.tasks.retrieve(task_id=task.id).video_id
    log.info(f"  Indexed {Path(sample.filepath).name} [{sample.label}] → {video_id}")
    return video_id


def fetch_embedding(client: TwelveLabs, index_id: str, video_id: str) -> list[float]:
    info = _rate_limited_call(
        client.indexes.videos.retrieve,
        _interval=10.0,
        index_id=index_id,
        video_id=video_id,
        embedding_option=["visual"],
    )
    segments = info.embedding.video_embedding.segments
    if len(segments) == 1:
        return segments[0].float_
    # Average all segment embeddings for multi-segment videos
    vecs = [np.array(seg.float_) for seg in segments]
    return np.mean(vecs, axis=0).tolist()


LABEL_PROMPT = """You are a worker-safety video classifier. Classify this video into exactly one label.

UNSAFE:
- Safe Walkway Violation
- Unauthorized Intervention
- Opened Panel Cover
- Carrying Overload with Forklift

SAFE:
- Safe Walkway
- Authorized Intervention
- Closed Panel Cover
- Safe Carrying

Return ONLY the exact label name. No punctuation, no explanation, no extra words."""


_last_call_time = 0.0


_consecutive_429s = 0


def _rate_limited_call(fn, *args, _max_retries=5, _interval=10.0, **kwargs):
    """Proactively throttle and retry on 429 with escalating backoff."""
    global _last_call_time, _consecutive_429s
    elapsed = time.time() - _last_call_time
    if elapsed < _interval:
        time.sleep(_interval - elapsed)

    for attempt in range(_max_retries):
        try:
            _last_call_time = time.time()
            result = fn(*args, **kwargs)
            _consecutive_429s = 0  # Reset on success
            return result
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "too_many_requests" in err_str:
                _consecutive_429s += 1
                # Parse retry-after header (seconds until quota resets)
                match = re.search(r"'retry-after':\s*'(\d+)'", err_str)
                retry_val = int(match.group(1)) if match else None
                # Check for daily request quota exhaustion
                remaining_match = re.search(r"'x-ratelimit-request-remaining':\s*'(\d+)'", err_str)
                remaining = int(remaining_match.group(1)) if remaining_match else None
                if remaining is not None and remaining == 0 and retry_val and retry_val > 300:
                    # Daily quota exhausted — no point retrying
                    from datetime import datetime, timezone
                    reset_time = datetime.fromtimestamp(time.time() + retry_val, tz=timezone.utc)
                    log.error(f"  Daily request quota exhausted (0 remaining). "
                              f"Resets at {reset_time.isoformat()}. Skipping remaining indexing.")
                    raise RuntimeError(
                        f"Daily request quota exhausted. Retry after {reset_time.isoformat()}"
                    )
                elif retry_val and retry_val < 300:
                    wait = retry_val + 3
                elif _consecutive_429s >= 3:
                    wait = 600
                    log.warning(f"  Persistent 429s detected ({_consecutive_429s} consecutive). "
                                f"Cooling down {wait}s...")
                else:
                    wait = 65
                log.info(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{_max_retries})")
                time.sleep(wait)
                _last_call_time = time.time()
            else:
                raise
    raise RuntimeError(f"Rate limit retries exhausted for {fn.__name__}")


def generate_label(client: TwelveLabs, video_id: str) -> str:
    result = _rate_limited_call(client.analyze, _interval=10.0, video_id=video_id, prompt=LABEL_PROMPT, temperature=0.1)
    return result.data.strip()


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def to_binary(label: str) -> str:
    return "safe" if label in SAFE_LABELS else "unsafe"


def bench_embedding_quality(samples: list[VideoSample]) -> dict:
    embeddings = np.array([s.embedding for s in samples])
    unique_labels = sorted({s.label for s in samples})
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[s.label] for s in samples])

    sil = silhouette_score(embeddings, y) if len(unique_labels) > 1 else 0.0

    # 8-class kNN
    k = min(5, len(samples) - 1)
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc_8 = cross_val_score(knn, embeddings, y, cv=cv, scoring="accuracy").mean()
    else:
        loo = LeaveOneOut()
        acc_8 = cross_val_score(knn, embeddings, y, cv=loo, scoring="accuracy").mean()

    # Binary kNN (safe vs unsafe)
    y_bin = np.array([0 if s.label in SAFE_LABELS else 1 for s in samples])
    knn_bin = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    n_splits_bin = min(5, min(np.bincount(y_bin)))
    if n_splits_bin >= 2:
        cv_bin = StratifiedKFold(n_splits=n_splits_bin, shuffle=True, random_state=42)
        acc_bin = cross_val_score(knn_bin, embeddings, y_bin, cv=cv_bin, scoring="accuracy").mean()
    else:
        acc_bin = 0.0

    return {"knn_accuracy": acc_8, "silhouette": sil, "binary_accuracy": acc_bin}


def bench_retrieval(client: TwelveLabs, index_id: str, samples: list[VideoSample]) -> dict:
    video_id_to_label = {s.marengo_video_id: s.label for s in samples}
    aps, recalls = [], []
    per_query = {}

    for query, expected_labels in RETRIEVAL_QUERIES.items():
        try:
            results = _rate_limited_call(
                client.search.query,
                _interval=10.0,
                index_id=index_id,
                query_text=query,
                search_options=["visual"],
            )
        except Exception as e:
            log.warning(f"Search failed for '{query}': {e}")
            continue

        hits = []
        count = 0
        for clip in results:
            if count >= 5:
                break
            lbl = video_id_to_label.get(clip.video_id)
            hits.append(1.0 if lbl in expected_labels else 0.0)
            count += 1

        if not hits:
            per_query[query] = {"ap": 0.0, "recall": 0.0}
            continue

        precisions, num_rel = [], 0
        for i, rel in enumerate(hits):
            num_rel += rel
            if rel:
                precisions.append(num_rel / (i + 1))
        ap = np.mean(precisions) if precisions else 0.0
        total_rel = sum(1 for s in samples if s.label in expected_labels)
        recall = num_rel / max(total_rel, 1)

        aps.append(ap)
        recalls.append(recall)
        per_query[query] = {"ap": round(ap, 3), "recall": round(recall, 3)}

    return {
        "map_at_5": np.mean(aps) if aps else 0.0,
        "recall_at_5": np.mean(recalls) if recalls else 0.0,
        "per_query": per_query,
    }


def bench_generation(samples: list[VideoSample]) -> dict:
    y_true = [s.label for s in samples]
    y_pred = [s.pegasus_label for s in samples]

    compliant = [1 if p in LABELS else 0 for p in y_pred]
    compliance_rate = np.mean(compliant)
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    accuracy = np.mean(correct)
    hallucination_rate = 1.0 - compliance_rate

    # Binary: safe vs unsafe
    y_true_bin = [to_binary(l) for l in y_true]
    y_pred_bin = [to_binary(l) if l in LABELS else "unknown" for l in y_pred]
    binary_correct = [1 if t == p else 0 for t, p in zip(y_true_bin, y_pred_bin)]
    binary_accuracy = np.mean(binary_correct)

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "accuracy": accuracy,
        "binary_accuracy": binary_accuracy,
        "format_compliance": compliance_rate,
        "hallucination_rate": hallucination_rate,
        "report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Twelve Labs safety eval harness")
    p.add_argument("--samples-per-label", type=int, default=5,
                    help="Videos to sample per label (default 5 = 40 total)")
    p.add_argument("--split", choices=["train", "test", "both"], default="test",
                    help="Dataset split to evaluate on (default: test)")
    p.add_argument("--fresh", action="store_true",
                    help="Ignore saved state and start from scratch")
    p.add_argument("--skip-indexing", action="store_true",
                    help="Skip indexing, run benchmarks on already-indexed videos")
    return p.parse_args()


def main():
    args = parse_args()
    client = TwelveLabs(api_key=API_KEY)

    # --- Load dataset ---
    samples = load_fiftyone_dataset(args.split, args.samples_per_label)
    if not args.fresh:
        samples = restore_state(samples)

    if not samples:
        raise SystemExit("No videos matched filters.")

    # --- Create separate indexes for each model ---
    marengo_index_id = get_or_create_index(client, MARENGO_INDEX_NAME, "marengo3.0")
    pegasus_index_id = get_or_create_index(client, PEGASUS_INDEX_NAME, "pegasus1.2")

    if not args.skip_indexing:
        # --- Step 1a: Index into Marengo index (duration-aware) ---
        log.info("=== Step 1a: Indexing videos into Marengo index ===")
        marengo_duration_used = sum(s.duration for s in samples if s.marengo_video_id)
        log.info(f"  Duration already indexed (Marengo): {marengo_duration_used:.1f}s / {MAX_DURATION_PER_INDEX:.0f}s")
        marengo_skipped = 0
        for i, s in enumerate(samples):
            if s.marengo_video_id:
                continue
            if marengo_duration_used + s.duration > MAX_DURATION_PER_INDEX:
                marengo_skipped += 1
                log.warning(f"  SKIP {Path(s.filepath).name} ({s.duration:.1f}s) — would exceed Marengo duration budget")
                continue
            n_indexed = sum(1 for x in samples if x.marengo_video_id)
            if n_indexed >= MAX_VIDEOS_PER_INDEX:
                log.warning(f"  SKIP — reached {MAX_VIDEOS_PER_INDEX} video limit for Marengo index")
                break
            try:
                s.marengo_video_id = index_video(client, marengo_index_id, s)
                marengo_duration_used += s.duration
                log.info(f"  Marengo duration used: {marengo_duration_used:.1f}s / {MAX_DURATION_PER_INDEX:.0f}s")
            except Exception as e:
                log.error(f"Failed to index {s.filepath} (Marengo): {e}")
            if (i + 1) % 5 == 0:
                save_state(samples)
        save_state(samples)

        marengo_indexed = [s for s in samples if s.marengo_video_id]
        log.info(f"Marengo indexed: {len(marengo_indexed)}/{len(samples)} "
                 f"({marengo_duration_used:.1f}s used, {marengo_skipped} skipped for budget)")

        # --- Step 1b: Index into Pegasus index (duration-aware) ---
        log.info("=== Step 1b: Indexing videos into Pegasus index ===")
        pegasus_duration_used = sum(s.duration for s in samples if s.pegasus_video_id)
        log.info(f"  Duration already indexed (Pegasus): {pegasus_duration_used:.1f}s / {MAX_DURATION_PER_INDEX:.0f}s")
        pegasus_skipped = 0
        for i, s in enumerate(samples):
            if s.pegasus_video_id:
                continue
            if pegasus_duration_used + s.duration > MAX_DURATION_PER_INDEX:
                pegasus_skipped += 1
                log.warning(f"  SKIP {Path(s.filepath).name} ({s.duration:.1f}s) — would exceed Pegasus duration budget")
                continue
            n_indexed = sum(1 for x in samples if x.pegasus_video_id)
            if n_indexed >= MAX_VIDEOS_PER_INDEX:
                log.warning(f"  SKIP — reached {MAX_VIDEOS_PER_INDEX} video limit for Pegasus index")
                break
            try:
                s.pegasus_video_id = index_video(client, pegasus_index_id, s)
                pegasus_duration_used += s.duration
                log.info(f"  Pegasus duration used: {pegasus_duration_used:.1f}s / {MAX_DURATION_PER_INDEX:.0f}s")
            except Exception as e:
                log.error(f"Failed to index {s.filepath} (Pegasus): {e}")
            if (i + 1) % 5 == 0:
                save_state(samples)
        save_state(samples)

        pegasus_indexed = [s for s in samples if s.pegasus_video_id]
        log.info(f"Pegasus indexed: {len(pegasus_indexed)}/{len(samples)} "
                 f"({pegasus_duration_used:.1f}s used, {pegasus_skipped} skipped for budget)")
    else:
        log.info("=== Skipping indexing (--skip-indexing) ===")
        marengo_indexed = [s for s in samples if s.marengo_video_id]
        pegasus_indexed = [s for s in samples if s.pegasus_video_id]

    marengo_duration_used = sum(s.duration for s in samples if s.marengo_video_id)
    pegasus_duration_used = sum(s.duration for s in samples if s.pegasus_video_id)
    marengo_indexed = [s for s in samples if s.marengo_video_id]
    pegasus_indexed = [s for s in samples if s.pegasus_video_id]

    total_duration_used = marengo_duration_used + pegasus_duration_used
    log.info(f"Total index usage: {total_duration_used:.1f}s / {13 * 60}s (13 min limit)")

    # --- Step 2: Embeddings (from Marengo index) ---
    log.info("=== Step 2: Fetching Marengo 3.0 embeddings ===")
    for s in marengo_indexed:
        if s.embedding:
            continue
        try:
            s.embedding = fetch_embedding(client, marengo_index_id, s.marengo_video_id)
        except Exception as e:
            log.error(f"Embedding fetch failed for {s.marengo_video_id}: {e}")

    with_emb = [s for s in marengo_indexed if s.embedding]
    log.info(f"Embeddings: {len(with_emb)}/{len(marengo_indexed)} "
             f"(dim={len(with_emb[0].embedding) if with_emb else '?'})")

    # --- Step 3: Pegasus labeling (from Pegasus index) ---
    log.info("=== Step 3: Pegasus 1.2 classification ===")
    for s in pegasus_indexed:
        if s.pegasus_label:
            continue
        try:
            s.pegasus_label = generate_label(client, s.pegasus_video_id)
            log.info(f"  GT={s.label:40s} → Pegasus='{s.pegasus_label}'")
        except Exception as e:
            log.error(f"Generation failed for {s.pegasus_video_id}: {e}")
            s.pegasus_label = "ERROR"
    save_state(samples)

    # --- Step 4: Benchmarks ---
    results = BenchmarkResults()

    if len(with_emb) >= 3:
        log.info("=== Benchmark A: Embedding quality ===")
        emb = bench_embedding_quality(with_emb)
        results.embedding_accuracy_knn = emb["knn_accuracy"]
        results.embedding_silhouette = emb["silhouette"]
        results.embedding_binary_accuracy = emb["binary_accuracy"]

    if RETRIEVAL_QUERIES:
        log.info("=== Benchmark B: Retrieval quality ===")
        ret = bench_retrieval(client, marengo_index_id, marengo_indexed)
        results.retrieval_map_at_5 = ret["map_at_5"]
        results.retrieval_recall_at_5 = ret["recall_at_5"]
        results.retrieval_per_query = ret["per_query"]

    labeled = [s for s in pegasus_indexed if s.pegasus_label and s.pegasus_label != "ERROR"]
    if labeled:
        log.info("=== Benchmark C: Generation quality ===")
        gen = bench_generation(labeled)
        results.generation_accuracy = gen["accuracy"]
        results.generation_binary_accuracy = gen["binary_accuracy"]
        results.generation_format_compliance = gen["format_compliance"]
        results.generation_hallucination_rate = gen["hallucination_rate"]
        results.per_class_report = gen["report"]
        results.confusion = gen["confusion_matrix"]

    print(results.summary())

    # Save JSON
    output = {
        "config": {
            "split": args.split,
            "samples_per_label": args.samples_per_label,
            "total_videos": len(samples),
            "marengo_indexed": len(marengo_indexed),
            "pegasus_indexed": len(pegasus_indexed),
            "marengo_duration_s": round(marengo_duration_used, 1),
            "pegasus_duration_s": round(pegasus_duration_used, 1),
            "total_index_usage_s": round(total_duration_used, 1),
            "dataset": HF_SLUG,
        },
        "samples": [
            {
                "filepath": s.filepath,
                "ground_truth": s.label,
                "duration": s.duration,
                "marengo_video_id": s.marengo_video_id,
                "pegasus_video_id": s.pegasus_video_id,
                "pegasus_label": s.pegasus_label,
                "embedding_dim": len(s.embedding) if s.embedding else 0,
            }
            for s in samples
        ],
        "metrics": {
            "embedding_knn_accuracy": results.embedding_accuracy_knn,
            "embedding_binary_accuracy": results.embedding_binary_accuracy,
            "embedding_silhouette": results.embedding_silhouette,
            "retrieval_map_at_5": results.retrieval_map_at_5,
            "retrieval_recall_at_5": results.retrieval_recall_at_5,
            "retrieval_per_query": results.retrieval_per_query,
            "generation_accuracy": results.generation_accuracy,
            "generation_binary_accuracy": results.generation_binary_accuracy,
            "generation_format_compliance": results.generation_format_compliance,
            "generation_hallucination_rate": results.generation_hallucination_rate,
        },
    }
    Path("eval_results.json").write_text(json.dumps(output, indent=2, default=str))
    log.info("Results saved to eval_results.json")


if __name__ == "__main__":
    main()
