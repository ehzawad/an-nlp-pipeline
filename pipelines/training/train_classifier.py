#!/usr/bin/env python3
"""Train classification model."""
import sys
import os
import csv
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, balanced_accuracy_score

# Disable tokenizers parallelism to avoid fork warnings with GridSearchCV
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def load_data(filepath: Path) -> Tuple[List[str], List[str]]:
    """Load questions and clusters from CSV."""
    questions, clusters = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['question'])
            clusters.append(row['cluster_name'])

    logger.info(f"Loaded {len(questions)} samples from {filepath}")
    return questions, clusters


def get_embeddings(model: SentenceTransformer, queries: List[str], prompt_template: str) -> np.ndarray:
    """Get embeddings for queries using E5 model."""
    # Format queries with prompt template
    formatted = [prompt_template.format(text=q) for q in queries]

    # Get embeddings (E5 normalizes by default)
    embeddings = model.encode(formatted, show_progress_bar=True, normalize_embeddings=True)

    logger.info(f"Generated embeddings: shape={embeddings.shape}")
    return embeddings


def train_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    config: dict
) -> LogisticRegression:
    """Train classifier with hyperparameter search and calibration."""
    model_config = config["model"]
    hp_config = config.get("hyperparameter_search", {})
    cal_config = config.get("calibration", {})

    # Base model
    base_model = LogisticRegression(
        solver=model_config.get("solver", "lbfgs"),
        max_iter=model_config.get("max_iter", 2000),
        class_weight=model_config.get("class_weight", "balanced"),
        random_state=model_config.get("random_state", 42)
    )

    # Hyperparameter search
    if hp_config.get("enabled", False):
        logger.info("Running hyperparameter search...")
        param_grid = {"C": hp_config.get("C_values", [0.25, 0.5, 1.0, 2.0])}

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=hp_config.get("cv_folds", 5),
            scoring=hp_config.get("scoring", "balanced_accuracy"),
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(train_embeddings, train_labels)

        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        classifier = grid_search.best_estimator_
    else:
        logger.info("Training without hyperparameter search...")
        classifier = base_model
        classifier.set_params(C=model_config.get("C", 1.0))
        classifier.fit(train_embeddings, train_labels)

    # Calibration
    if cal_config.get("enabled", False):
        logger.info("Calibrating classifier...")
        classifier = CalibratedClassifierCV(
            classifier,
            method=cal_config.get("method", "sigmoid"),
            cv=cal_config.get("cv", 3)
        )
        classifier.fit(train_embeddings, train_labels)
        logger.info("Calibration complete")

    return classifier


def evaluate_model(
    classifier: LogisticRegression,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    label_encoder: LabelEncoder
) -> dict:
    """Evaluate model on validation set."""
    logger.info("Evaluating on validation set...")

    # Predictions
    predictions = classifier.predict(val_embeddings)

    # Metrics
    accuracy = balanced_accuracy_score(val_labels, predictions)
    report = classification_report(
        val_labels,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    logger.info(f"Validation Balanced Accuracy: {accuracy:.4f}")

    return {
        "balanced_accuracy": accuracy,
        "classification_report": report
    }


def main():
    """Main training function."""
    print("=" * 70)
    print("Classification Model Training")
    print("=" * 70)

    # Load configs
    config_loader = ConfigLoader()
    main_config = config_loader.load("main")
    class_config = config_loader.load("classification")

    # Load E5 model
    logger.info("Loading E5 model...")
    e5_model = SentenceTransformer(
        main_config["models"]["e5"]["name"],
        cache_folder=str(PROJECT_ROOT / main_config["models"]["e5"]["cache_dir"]),
        device=main_config["models"]["e5"]["device"]
    )
    logger.info("E5 model loaded")

    # Load datasets
    datasets_dir = PROJECT_ROOT / main_config["paths"]["datasets_processed"]
    train_file = datasets_dir / class_config["artifacts"]["train_file"]
    val_file = datasets_dir / class_config["artifacts"]["val_file"]

    train_questions, train_clusters = load_data(train_file)
    val_questions, val_clusters = load_data(val_file)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_clusters)
    val_labels = label_encoder.transform(val_clusters)

    logger.info(f"Number of classes: {len(label_encoder.classes_)}")

    # Get embeddings
    prompt_template = class_config.get("prompt_template", "query: {text}")
    logger.info("Generating training embeddings...")
    train_embeddings = get_embeddings(e5_model, train_questions, prompt_template)

    logger.info("Generating validation embeddings...")
    val_embeddings = get_embeddings(e5_model, val_questions, prompt_template)

    # Train classifier
    logger.info("Training classifier...")
    classifier = train_classifier(train_embeddings, train_labels, class_config)
    logger.info("Training complete")

    # Evaluate
    metrics = evaluate_model(classifier, val_embeddings, val_labels, label_encoder)

    # Save artifacts
    models_dir = PROJECT_ROOT / main_config["paths"]["models"] / class_config["artifacts"]["model_subdir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pkl"
    label_encoder_path = models_dir / "label_encoder.pkl"
    metrics_path = models_dir / "validation_metrics.json"

    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved model to {model_path}")

    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Saved label encoder to {label_encoder_path}")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    print("\n" + "=" * 70)
    print(f"✓ Training complete! Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
