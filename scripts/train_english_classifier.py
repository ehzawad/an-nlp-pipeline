#!/usr/bin/env python3
"""
Quick training script for English test dataset.
Uses the existing training infrastructure but with a different dataset.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import random
from collections import Counter
from src.components.classification.classifier import Classifier
from src.core.logger import logger
from src.core.config_loader import config_loader
from src.core.model_registry import registry

def main():
    print("=" * 70)
    print("Training Classifier on English Test Dataset")
    print("=" * 70)

    # Load English test dataset
    questions = []
    clusters = []
    tags = []

    with open("datasets/raw/test_english_dataset.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["question"]:  # Skip empty rows
                questions.append(row["question"])
                clusters.append(row["cluster"])
                tags.append(row["tag"])

    print(f"\n✓ Loaded {len(questions)} questions")
    print(f"✓ Clusters: {list(set(clusters))}")
    print(f"✓ Tags: {list(set(tags))}")

    # Split dataset (80/10/10)
    indices = list(range(len(questions)))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_questions = [questions[i] for i in train_indices]
    train_clusters = [clusters[i] for i in train_indices]

    val_questions = [questions[i] for i in val_indices]
    val_clusters = [clusters[i] for i in val_indices]

    test_questions = [questions[i] for i in test_indices]
    test_clusters = [clusters[i] for i in test_indices]

    print(f"\n✓ Train: {len(train_questions)} samples")
    print(f"✓ Val: {len(val_questions)} samples")
    print(f"✓ Test: {len(test_questions)} samples")

    # Initialize classifier
    config = config_loader.load("classification")
    classifier = Classifier(config)

    # Register and load E5 model
    print("\n✓ Loading E5 embedding model...")
    from src.models.embeddings import E5EmbeddingModel
    main_config = config_loader.load("main")
    e5_config = main_config["models"]["e5"]
    e5_model = E5EmbeddingModel(
        model_name=e5_config["name"],
        cache_dir=e5_config["cache_dir"]
    )
    registry.register("e5", e5_model, lazy=False)
    embedding_model = registry.get("e5")

    # Train
    print("\n✓ Training classifier...")
    metrics = classifier.train(
        questions=train_questions,
        labels=train_clusters,
        val_questions=val_questions,
        val_labels=val_clusters
    )

    # Save model
    output_dir = "models/classification_english"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    classifier.save(output_dir)
    print(f"\n✓ Saved classifier to {output_dir}")

    # Print metrics
    print("\n" + "=" * 70)
    print("Training Results:")
    print("=" * 70)
    print(f"Validation Accuracy: {metrics['val_accuracy']*100:.2f}%")
    print(f"Validation Balanced Accuracy: {metrics['val_balanced_accuracy']*100:.2f}%")

    # Test
    print("\n✓ Testing on holdout set...")
    test_predictions = []
    for question in test_questions:
        pred = classifier.predict(question, top_k=1)
        test_predictions.append(pred[0]["cluster"])

    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    test_accuracy = accuracy_score(test_clusters, test_predictions)
    test_balanced = balanced_accuracy_score(test_clusters, test_predictions)

    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_balanced*100:.2f}%")

    print("\n" + "=" * 70)
    print(" English classifier training complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
