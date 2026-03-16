#!/usr/bin/env python3
"""
Visual AI Hackathon Worker Safety Challenge Enablement Kit
==========================================================

This script demonstrates an end-to-end workflow between TwelveLabs and FiftyOne
for semantic dataset curation and visualization in the workplace safety domain.

Key Features:
- Video ingestion and indexing via TwelveLabs Marengo 3.0
- Semantic clustering using video embeddings
- Auto-labeling with TwelveLabs Pegasus 1.2
- Interactive visualization with FiftyOne
- Export to PyTorch using FiftyOne's to_torch() pattern

Usage:
    1. Set environment variables in .env file
    2. Run: python main.py
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from dotenv import load_dotenv

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F
from fiftyone.core.labels import Classification
from fiftyone.utils.torch import GetItem
from fiftyone.utils.huggingface import load_from_hub

from twelvelabs import TwelveLabs
from twelvelabs.indexes import IndexesCreateRequestModelsItem


# =============================================================================
# Configuration
# =============================================================================

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()

    config = {
        "dataset_name": os.getenv("DATASET_NAME", "safe_unsafe_behaviours"),
        "dataset_split": os.getenv("DATASET_SPLIT", "train"),
        "videos_per_label": int(os.getenv("VIDEOS_PER_LABEL", "3")),
        "min_duration": float(os.getenv("MIN_DURATION", "4.0")),
        "num_clusters": int(os.getenv("NUM_CLUSTERS", "8")),
        "tl_index_name": os.getenv("TL_INDEX_NAME", "fiftyone-twelvelabs-index"),
        "tl_api_key": os.getenv("TL_API_KEY"),
    }

    if not config["tl_api_key"]:
        raise ValueError("TL_API_KEY must be set in the .env file")

    return config


# Label mapping for the Worker Safety dataset
LABEL_TO_IDX = {
    "Safe Walkway Violation": 0,
    "Unauthorized Intervention": 1,
    "Opened Panel Cover": 2,
    "Carrying Overload with Forklift": 3,
    "Safe Walkway": 4,
    "Authorized Intervention": 5,
    "Closed Panel Cover": 6,
    "Safe Carrying": 7,
}

# Prompt for Pegasus auto-labeling
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


# =============================================================================
# TwelveLabs Integration
# =============================================================================

def get_or_create_twelvelabs_index(client: TwelveLabs, index_name: str) -> str:
    """
    Get existing TwelveLabs index or create a new one.

    Args:
        client: TwelveLabs client instance
        index_name: Name of the index to retrieve or create

    Returns:
        Index ID
    """
    indexes = client.indexes.list()
    for index in indexes:
        if index.index_name == index_name:
            print(f"Found index '{index_name}' with ID {index.id}")
            return index.id

    # Create new index with Marengo 3.0 and Pegasus 1.2
    index = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="marengo3.0", model_options=["visual", "audio"]
            ),
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2", model_options=["visual", "audio"]
            ),
        ],
    )
    print(f"Created index '{index_name}' with ID {index.id}")
    return index.id


def index_video_to_twelvelabs(
    client: TwelveLabs, index_id: str, sample: fo.Sample
) -> str:
    """
    Upload a FiftyOne video sample to TwelveLabs for indexing.

    This function reads the video file from disk, uploads it to TwelveLabs,
    and waits for the indexing task to complete.

    Args:
        client: TwelveLabs client instance
        index_id: TwelveLabs index ID
        sample: A FiftyOne Sample object

    Returns:
        The TwelveLabs video_id assigned to the indexed video.

    Raises:
        Exception: If the indexing task fails.
    """
    with open(sample.filepath, "rb") as f:
        task = client.tasks.create(
            index_id=index_id,
            video_file=f.read(),
            user_metadata=json.dumps({
                "filepath": sample.filepath,
                "sample_id": sample.id,
                "label": sample.ground_truth.label if sample.ground_truth else None,
            }),
        )

    task = client.tasks.wait_for_done(task_id=task.id)
    if task.status != "ready":
        raise Exception(f"Task failed: {task.status}")

    return client.tasks.retrieve(task_id=task.id).video_id


def fetch_embedding(client: TwelveLabs, index_id: str, video_id: str) -> list:
    """Fetch visual embedding for a TwelveLabs video."""
    video_info = client.indexes.videos.retrieve(
        index_id=index_id,
        video_id=video_id,
        embedding_option=["visual"],
    )
    return video_info.embedding.video_embedding.segments[0].float_


def generate_label(client: TwelveLabs, video_id: str) -> str:
    """
    Generate a semantic label for a video using TwelveLabs Pegasus 1.2.

    Args:
        client: TwelveLabs client instance
        video_id: TwelveLabs video ID

    Returns:
        Generated label string
    """
    result = client.analyze(
        video_id=video_id,
        prompt=CLUSTER_LABEL_PROMPT,
        temperature=0.2,
    )
    return result.data


# =============================================================================
# FiftyOne Dataset Operations
# =============================================================================

def load_or_create_dataset(dataset_name: str) -> fo.Dataset:
    """Load existing dataset or create from Hugging Face."""
    if fo.dataset_exists(dataset_name):
        print(f"Loading existing dataset '{dataset_name}'...")
        return fo.load_dataset(dataset_name)

    print(f"Downloading dataset from Hugging Face...")
    dataset = load_from_hub(
        "Voxel51/Safe_and_Unsafe_Behaviours",
        name=dataset_name,
        persistent=True,
        overwrite=False,
    )
    return dataset


def ingest_videos(
    client: TwelveLabs,
    index_id: str,
    dataset: fo.Dataset,
    dataset_split: str,
    videos_per_label: int,
    min_duration: float,
) -> int:
    """
    Ingest videos from FiftyOne dataset into TwelveLabs index.

    Uses FiftyOne views to filter and iterate efficiently:
    - Filters by split tag and minimum duration
    - Skips already-indexed videos (idempotent)
    - Limits to videos_per_label per class

    Args:
        client: TwelveLabs client instance
        index_id: TwelveLabs index ID
        dataset: FiftyOne dataset
        dataset_split: Split to process (e.g., "train")
        videos_per_label: Max videos per label
        min_duration: Minimum video duration in seconds

    Returns:
        Number of newly indexed videos
    """
    # Build filtered view using FiftyOne's view operations
    base_view = (
        dataset
        .match_tags(dataset_split)
        .match(F("metadata.duration") >= min_duration)
        .match(~F("tl_video_id").exists())  # Skip already indexed
    )

    indexed_count = 0

    # Process each label separately for balanced sampling
    for label in base_view.distinct("ground_truth.label"):
        label_view = (
            base_view
            .match(F("ground_truth.label") == label)
            .take(videos_per_label)
        )
        print(f"\n{label}: {len(label_view)} to index")

        for sample in label_view.iter_samples(autosave=True, progress=True):
            try:
                sample["tl_video_id"] = index_video_to_twelvelabs(
                    client, index_id, sample
                )
                print(f"  ✓ {sample.filename}")
                indexed_count += 1
            except Exception as e:
                print(f"  ✗ {sample.filename}: {e}")

    print(f"\nTotal indexed: {len(dataset.exists('tl_video_id'))}")
    return indexed_count


def fetch_embeddings(
    client: TwelveLabs,
    index_id: str,
    dataset: fo.Dataset,
) -> list:
    """
    Fetch embeddings from TwelveLabs and store on FiftyOne samples.

    Uses efficient FiftyOne patterns:
    - select_fields() to load only needed data
    - iter_samples(autosave=True) for batched writes

    Returns:
        List of embeddings (for clustering)
    """
    indexed_view = dataset.exists("tl_video_id")
    print(f"Fetching embeddings for {len(indexed_view)} indexed videos...")

    embeddings = []
    for sample in indexed_view.select_fields("tl_video_id").iter_samples(
        progress=True, autosave=True
    ):
        embedding = fetch_embedding(client, index_id, sample.tl_video_id)
        sample["tl_embedding"] = embedding
        embeddings.append(embedding)

    print(f"\nStored {len(embeddings)} embeddings on dataset")
    return embeddings


def cluster_and_label(
    client: TwelveLabs,
    dataset: fo.Dataset,
    embeddings: list,
    num_clusters: int = 8,
) -> tuple:
    """
    Cluster videos and generate semantic labels using Pegasus.

    Uses efficient FiftyOne patterns:
    - values() to extract fields as lists
    - set_values() for batch updates

    Returns:
        Tuple of (cluster_labels array, cluster_label_map dict)
    """
    print(f"Clustering {len(embeddings)} videos into {num_clusters} clusters...")

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Get indexed view and extract video IDs
    indexed_view = dataset.exists("tl_video_id")
    video_ids = indexed_view.values("tl_video_id")

    # Generate labels for each cluster (one API call per cluster)
    cluster_label_map = {}
    cluster_strings = []

    for video_id, cluster_idx in zip(video_ids, cluster_labels):
        if cluster_idx not in cluster_label_map:
            print(f"Generating label for cluster {cluster_idx}...")
            cluster_label_map[cluster_idx] = generate_label(client, video_id)
            print(f"  -> {cluster_label_map[cluster_idx]}")
        cluster_strings.append(cluster_label_map[cluster_idx])

    # Batch update using set_values with Classification objects
    classifications = [Classification(label=s) for s in cluster_strings]
    indexed_view.set_values("pred_cluster", classifications)

    print(f"\nCluster labels: {cluster_label_map}")
    return cluster_labels, cluster_label_map


def compute_visualization(dataset: fo.Dataset):
    """Compute 2D UMAP visualization of embeddings."""
    print("Computing 2D visualization with UMAP...")

    indexed_view = dataset.exists("tl_video_id")

    results = fob.compute_visualization(
        indexed_view,
        embeddings="tl_embedding",
        num_dims=2,
        brain_key="tl_embeddings_viz",
        method="umap",
        verbose=True,
        seed=51,
    )
    return results


# =============================================================================
# PyTorch Export using GetItem Pattern
# =============================================================================

class WorkerSafetyGetItem(GetItem):
    """
    Extracts embeddings and labels from FiftyOne samples for classifier training.

    Transforms each sample into:
    - embedding: 512-dim tensor from TwelveLabs Marengo
    - label_idx: integer class index
    - label_str: human-readable label string
    """

    def __init__(self, label_to_idx, field_mapping=None):
        self.label_to_idx = label_to_idx
        super().__init__(field_mapping=field_mapping)

    @property
    def required_keys(self):
        return ["tl_embedding", "ground_truth"]

    def __call__(self, d):
        embedding = d.get("tl_embedding")
        ground_truth = d.get("ground_truth")
        label_str = ground_truth.label if ground_truth else "Unknown"

        return {
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "label_idx": torch.tensor(
                self.label_to_idx.get(label_str, -1), dtype=torch.long
            ),
            "label_str": label_str,
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        "embedding": torch.stack([b["embedding"] for b in batch]),
        "label_idx": torch.stack([b["label_idx"] for b in batch]),
        "label_str": [b["label_str"] for b in batch],
    }


def create_dataloader(dataset: fo.Dataset, batch_size: int = 4) -> DataLoader:
    """
    Create PyTorch DataLoader from FiftyOne dataset using to_torch().

    Args:
        dataset: FiftyOne dataset with tl_embedding field
        batch_size: Batch size for DataLoader

    Returns:
        PyTorch DataLoader ready for training
    """
    indexed_view = dataset.exists("tl_video_id")

    getter = WorkerSafetyGetItem(LABEL_TO_IDX)
    torch_dataset = indexed_view.to_torch(getter)

    print(f"Created PyTorch dataset with {len(torch_dataset)} samples")

    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return dataloader


def export_dataloader(dataloader: DataLoader, output_path: str):
    """Export DataLoader info for verification."""
    batch = next(iter(dataloader))
    print(f"Batch embeddings shape: {batch['embedding'].shape}")
    print(f"Batch labels: {batch['label_idx']}")
    print(f"Export complete. Use create_dataloader() in your training script.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the enablement kit."""
    parser = argparse.ArgumentParser(
        description="CVPR 2026 Worker Safety Challenge Enablement Kit"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip video ingestion (use existing indexed videos)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering and labeling",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only create DataLoader (skip visualization)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader (default: 4)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize TwelveLabs client
    print("Initializing TwelveLabs client...")
    tl_client = TwelveLabs(api_key=config["tl_api_key"])

    # Get or create TwelveLabs index
    index_id = get_or_create_twelvelabs_index(tl_client, config["tl_index_name"])

    # Load or create FiftyOne dataset
    dataset = load_or_create_dataset(config["dataset_name"])

    # Video ingestion (optional)
    if not args.skip_ingestion:
        ingest_videos(
            client=tl_client,
            index_id=index_id,
            dataset=dataset,
            dataset_split=config["dataset_split"],
            videos_per_label=config["videos_per_label"],
            min_duration=config["min_duration"],
        )

    # Fetch embeddings
    indexed_view = dataset.exists("tl_video_id")
    if len(indexed_view) == 0:
        print("No indexed videos found. Run without --skip-ingestion first.")
        return

    # Check if embeddings already exist
    if not indexed_view.first().has_field("tl_embedding"):
        embeddings = fetch_embeddings(tl_client, index_id, dataset)
    else:
        print("Embeddings already exist, loading from dataset...")
        embeddings = indexed_view.values("tl_embedding")

    # Cluster and auto-label videos
    if not args.skip_clustering:
        cluster_labels, label_map = cluster_and_label(
            client=tl_client,
            dataset=dataset,
            embeddings=embeddings,
            num_clusters=config["num_clusters"],
        )

    # Create PyTorch DataLoader
    dataloader = create_dataloader(dataset, batch_size=args.batch_size)
    export_dataloader(dataloader, "worker_safety_dataset")

    # Launch visualization (optional)
    if not args.export_only:
        compute_visualization(dataset)

        print("\nLaunching FiftyOne App...")
        print("Navigate to http://localhost:5151 in your browser")
        session = fo.launch_app(dataset, auto=False, port=5151)
        session.wait()


if __name__ == "__main__":
    main()
