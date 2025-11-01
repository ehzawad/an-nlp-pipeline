#!/usr/bin/env python3
"""Build FAISS indices for semantic search."""
import sys
import csv
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import config loader
from src.application.config.loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_questions_with_clusters(filepath: Path) -> List[Dict[str, str]]:
    """Load questions with cluster information."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "question": row['question'],
                "cluster": row['cluster_name']
            })

    logger.info(f"Loaded {len(data)} questions from {filepath}")
    return data


def load_tags_mapping(filepath: Path) -> Dict[str, str]:
    """Load question to tag mapping."""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['question']] = row.get('tag', '')

    logger.info(f"Loaded {len(mapping)} tag mappings from {filepath}")
    return mapping


def build_indices(
    questions_data: List[Dict[str, str]],
    embeddings: np.ndarray,
    output_dir: Path,
    config: dict
):
    """Build per-cluster and merged FAISS indices."""
    faiss_config = config.get("faiss", {})
    normalize = faiss_config.get("normalize_embeddings", True)

    # Normalize embeddings if needed
    if normalize:
        faiss.normalize_L2(embeddings)
        logger.info("Normalized embeddings for cosine similarity")

    # Group questions by cluster
    cluster_groups = defaultdict(list)
    for idx, item in enumerate(questions_data):
        cluster_groups[item['cluster']].append(idx)

    logger.info(f"Found {len(cluster_groups)} clusters")

    # Create indices directory
    indices_dir = output_dir / faiss_config.get("indices_subdir", "faiss_indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    # Build per-cluster indices
    logger.info("Building per-cluster indices...")
    for cluster, indices in cluster_groups.items():
        cluster_embeddings = embeddings[indices]

        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(cluster_embeddings)

        # Save index
        index_path = indices_dir / f"{cluster}.index"
        faiss.write_index(index, str(index_path))
        logger.info(f"  {cluster}: {len(indices)} questions -> {index_path}")

    # Build merged index if enabled
    if faiss_config.get("build_merged_index", True):
        logger.info("Building merged index...")
        merged_index = faiss.IndexFlatIP(embeddings.shape[1])
        merged_index.add(embeddings)

        merged_path = indices_dir / "merged_all.index"
        faiss.write_index(merged_index, str(merged_path))
        logger.info(f"  merged_all: {len(embeddings)} questions -> {merged_path}")

    # Save metadata
    metadata_path = output_dir / "cluster_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main indexing function."""
    print("=" * 70)
    print("FAISS Index Building")
    print("=" * 70)

    # Load configs
    config_loader = ConfigLoader()
    main_config = config_loader.load("main")
    search_config = config_loader.load("semantic_search")

    # Load E5 model
    logger.info("Loading E5 model...")
    e5_model = SentenceTransformer(
        main_config["models"]["e5"]["name"],
        cache_folder=str(PROJECT_ROOT / main_config["models"]["e5"]["cache_dir"]),
        device=main_config["models"]["e5"]["device"]
    )
    logger.info("E5 model loaded")

    # Load data
    datasets_dir = PROJECT_ROOT / main_config["paths"]["datasets_raw"]
    artifacts = search_config.get("artifacts", {})

    cluster_file = datasets_dir / artifacts.get("cluster_file", "questions_with_clusters.csv")
    tags_file = datasets_dir / artifacts.get("tags_file", "questions_with_tags.csv")

    questions_data = load_questions_with_clusters(cluster_file)
    tags_mapping = load_tags_mapping(tags_file)

    # Add tags to questions_data
    for item in questions_data:
        item['tag'] = tags_mapping.get(item['question'], '')

    # Generate embeddings
    logger.info("Generating embeddings...")
    prompt_template = search_config.get("prompt_template", "query: {text}")
    questions_list = [item['question'] for item in questions_data]
    formatted_queries = [prompt_template.format(text=q) for q in questions_list]

    embeddings = e5_model.encode(
        formatted_queries,
        show_progress_bar=search_config.get("indexing", {}).get("show_progress", True),
        normalize_embeddings=search_config.get("faiss", {}).get("normalize_embeddings", True)
    )
    logger.info(f"Generated embeddings: shape={embeddings.shape}")

    # Build indices
    output_dir = PROJECT_ROOT / main_config["paths"]["models"] / artifacts.get("model_subdir", "semantic_search")
    build_indices(questions_data, embeddings, output_dir, search_config)

    print("\n" + "=" * 70)
    print("✓ Index building complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
