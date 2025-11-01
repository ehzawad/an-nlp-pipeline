#!/usr/bin/env python3
"""Train binary classifier to detect fractional (incomplete) queries."""

import sys
import json
import pickle
import csv
import logging
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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


def load_fractional_queries(filepath: Path) -> list:
    """Load fractional query examples from text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} fractional queries from {filepath}")
    return queries


def load_complete_queries(filepath: Path, max_samples: int = None) -> list:
    """Load complete queries from dataset CSV."""
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get('question', '').strip()
            if question:
                queries.append(question)

            if max_samples and len(queries) >= max_samples:
                break

    logger.info(f"Loaded {len(queries)} complete queries from {filepath}")
    return queries


def prepare_dataset(fractional_queries: list, complete_queries: list):
    """
    Prepare training dataset with balanced classes.

    Args:
        fractional_queries: List of incomplete/fractional queries (positive class)
        complete_queries: List of complete queries (negative class)

    Returns:
        X (texts), y (labels)
    """
    # Balance classes
    min_samples = min(len(fractional_queries), len(complete_queries))

    # Sample equal amounts
    fractional_sample = fractional_queries[:min_samples]
    complete_sample = complete_queries[:min_samples]

    # Create dataset
    X = fractional_sample + complete_sample
    y = [1] * len(fractional_sample) + [0] * len(complete_sample)

    logger.info(f"Prepared balanced dataset: {min_samples} fractional, {min_samples} complete")

    return X, y


def encode_queries(queries: list, model: SentenceTransformer) -> np.ndarray:
    """Encode queries using E5 model."""
    logger.info(f"Encoding {len(queries)} queries with E5...")

    # E5 requires "query: " prefix for queries
    queries_with_prefix = [f"query: {q}" for q in queries]

    embeddings = model.encode(
        queries_with_prefix,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    logger.info(f"Encoded to shape: {embeddings.shape}")
    return embeddings


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Train logistic regression classifier."""
    logger.info("Training Logistic Regression classifier...")

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )

    clf.fit(X_train, y_train)

    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")

    # Predictions and detailed metrics
    y_pred = clf.predict(X_test)

    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Complete', 'Fractional']))

    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return clf


def save_classifier(classifier, output_path: Path, metadata: dict):
    """Save classifier and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save classifier
    with open(output_path, 'wb') as f:
        pickle.dump(classifier, f)

    logger.info(f"Classifier saved to: {output_path}")

    # Save metadata
    metadata_path = output_path.with_name(output_path.stem + '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Training Fractional Query Classifier")
    print("=" * 70)

    # Load config
    config_loader = ConfigLoader()
    main_config = config_loader.load("main")

    # Load E5 model
    logger.info("Loading E5 model...")
    e5_model = SentenceTransformer(
        main_config["models"]["e5"]["name"],
        cache_folder=str(PROJECT_ROOT / main_config["models"]["e5"]["cache_dir"]),
        device=main_config["models"]["e5"]["device"]
    )
    logger.info("E5 model loaded")

    # Paths
    fractional_file = PROJECT_ROOT / "datasets/raw/fractional_queries.txt"
    complete_file = PROJECT_ROOT / "datasets/raw/questions_with_tags.csv"
    output_path = PROJECT_ROOT / main_config["paths"]["models"] / "context" / "fractional_classifier.pkl"

    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("Loading Data")
    logger.info("=" * 70)

    fractional_queries = load_fractional_queries(fractional_file)
    complete_queries = load_complete_queries(complete_file)

    # Prepare dataset
    X_text, y = prepare_dataset(fractional_queries, complete_queries)

    # Encode with E5
    logger.info("\n" + "=" * 70)
    logger.info("Encoding Queries")
    logger.info("=" * 70)

    X_embeddings = encode_queries(X_text, e5_model)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train classifier
    logger.info("\n" + "=" * 70)
    logger.info("Training Classifier")
    logger.info("=" * 70)

    classifier = train_classifier(X_train, y_train, X_test, y_test)

    # Save
    logger.info("\n" + "=" * 70)
    logger.info("Saving Model")
    logger.info("=" * 70)

    metadata = {
        "model_type": "LogisticRegression",
        "embedding_model": main_config["models"]["e5"]["name"],
        "n_fractional_samples": len(fractional_queries),
        "n_complete_samples": len(complete_queries),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": float(classifier.score(X_train, y_train)),
        "test_accuracy": float(classifier.score(X_test, y_test))
    }

    save_classifier(classifier, output_path, metadata)

    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
