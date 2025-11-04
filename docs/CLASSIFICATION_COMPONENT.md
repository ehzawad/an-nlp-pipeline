# Classification Component Documentation

## Overview

The classification component is a core part of the NLP pipeline that performs intent classification on user queries. It uses a 48-class logistic regression classifier trained on multilingual E5 embeddings to categorize Bengali dialogue utterances into support clusters.

---

## Component Location

### Core Implementation

**Primary File:** `src/application/models/classifiers.py`

This file contains the `ClassifierModel` class which implements the intent classifier with the following key features:

- **48-class intent classification** using logistic regression
- **E5 embedding integration** for feature extraction
- **Top-k prediction** with confidence scores
- **Lazy loading** of model artifacts
- **Thread-safe operations** for async/concurrent usage

**Key Methods:**
```python
def predict(query: str, top_k: int = 3) -> List[Dict[str, Any]]
    # Returns top-k predictions with cluster names and confidence scores

def predict_single(query: str) -> Dict[str, Any]
    # Returns single top prediction
```

### NLP Pipeline Integration

**File:** `src/application/nlp_service.py`

The classifier is integrated into the `AsyncNLPPipeline` class which:

- Runs classification in a thread pool (non-blocking async operation)
- Uses classification confidence to determine search strategy
- Implements circuit breaker pattern for fault tolerance
- Provides tracing and observability hooks

**Pipeline Flow:**
1. Query is classified (async, in thread pool)
2. Top confidence score is evaluated against threshold
3. If confidence ≥ threshold → search in predicted cluster
4. If confidence < threshold → search in merged index

### Training Scripts

**File:** `pipelines/training/train_classifier.py`

Complete training pipeline that:

- Loads stratified train/validation splits
- Generates E5 embeddings for text queries
- Trains logistic regression with optional:
  - Grid search for hyperparameter tuning (C values)
  - Probability calibration (sigmoid or isotonic)
- Evaluates on validation set with balanced accuracy
- Saves model artifacts (pickle files) and metrics

**Usage:**
```bash
cd /home/runner/work/an-nlp-pipeline/an-nlp-pipeline
python pipelines/training/train_classifier.py
```

**Output Artifacts:**
- `models/classification/model.pkl` - Trained classifier
- `models/classification/label_encoder.pkl` - Label encoder for 48 classes
- `models/classification/validation_metrics.json` - Performance metrics

---

## Configuration

**File:** `config/classification.json`

Contains all classifier configuration including:

```json
{
  "model": {
    "type": "logistic_regression",
    "C": 0.25,
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": 42
  },
  "hyperparameter_search": {
    "enabled": true,
    "C_values": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "cv_folds": 5,
    "scoring": "balanced_accuracy"
  },
  "calibration": {
    "enabled": true,
    "method": "sigmoid",
    "cv": 3
  },
  "prompt_template": "task: classification | query: {text}",
  "artifacts": {
    "model_subdir": "classification",
    "train_file": "classification/train.csv",
    "val_file": "classification/val.csv"
  },
  "num_classes": 48
}
```

**Additional Configuration:**
- `config/main.json` - E5 embedding model settings, paths
- `config/dialogue.json` - Confidence thresholds for classification-based routing

---

## Data Pipeline

### Dataset Preparation

**File:** `pipelines/data/prepare_datasets.py`

Prepares stratified train/validation/test splits from raw data.

### Raw Data

**Location:** `datasets/raw/`

- `questions_with_clusters.csv` - 3,200+ Bengali utterances with cluster labels
- Contains the training data for the 48-class classifier

### Processed Data

**Location:** `datasets/processed/classification/`

After running `prepare_datasets.py`:
- `train.csv` - Training set
- `val.csv` - Validation set
- `test.csv` - Test set

All splits are stratified to maintain class distribution.

---

## Model Artifacts

**Location:** `models/classification/`

Contains the following after training:

1. **model.pkl** - Scikit-learn LogisticRegression classifier
   - 48 output classes (support clusters)
   - Trained on E5 embeddings (1024-dimensional)
   - Balanced class weights

2. **label_encoder.pkl** - LabelEncoder instance
   - Maps cluster names ↔ numeric indices
   - Used for encoding/decoding predictions

3. **validation_metrics.json** - Performance metrics
   - Balanced accuracy (~0.91 for current model)
   - Per-class precision/recall/f1-scores
   - Classification report

---

## Integration with Other Components

### 1. Embedding Model

**File:** `src/application/models/embeddings.py`

The classifier depends on `E5EmbeddingModel` for feature extraction:

```python
embedding_model = E5EmbeddingModel(config)
classifier = ClassifierModel(config, embedding_model)
```

### 2. Tenant Context

**File:** `src/shared/tenant_context.py`

The classifier is lazily initialized per tenant:

```python
# Lazy initialization in TenantContext
classifier = ClassifierModel(classifier_config, embedding_model)
classifier.load()  # Loads pickled artifacts
```

### 3. Circuit Breaker

**File:** `src/infrastructure/circuit_breaker.py`

Classification operations are protected by circuit breaker to prevent cascading failures.

### 4. Tracing

**File:** `src/infrastructure/tracing.py`

Classification operations are traced with:
- Operation duration
- Query metadata
- Prediction confidence
- Error information

---

## Usage Examples

### Basic Classification

```python
from src.application.models import ClassifierModel, E5EmbeddingModel, ModelConfig

# Initialize embedding model with ModelConfig
embedding_config = ModelConfig(
    name="e5_embeddings",
    type="embeddings",
    implementation="E5EmbeddingModel",
    config={
        "name": "intfloat/multilingual-e5-large-instruct",
        "cache_dir": "./models/embeddings/e5_cache",
        "device": "cpu"
    }
)
embedding_model = E5EmbeddingModel(embedding_config)

# Initialize classifier with ModelConfig
classifier_config = ModelConfig(
    name="classifier",
    type="classifier",
    implementation="ClassifierModel",
    config={
        "model_path": "models/classification/model.pkl",
        "label_encoder_path": "models/classification/label_encoder.pkl"
    }
)
classifier = ClassifierModel(classifier_config, embedding_model)

# Predict
results = classifier.predict("আমার NID কার্ড হারিয়ে গেছে", top_k=3)
# Returns:
# [
#   {"cluster": "lost_nid", "confidence": 0.85},
#   {"cluster": "nid_replacement", "confidence": 0.10},
#   {"cluster": "nid_status", "confidence": 0.03}
# ]
```

**Note:** In practice, models are typically initialized via `TenantContext` which handles configuration loading automatically.

### Via NLP Pipeline

```python
from src.application.nlp_service import AsyncNLPPipeline

# Assume classifier and searcher are already initialized
# (typically through TenantContext)

# Initialize pipeline
pipeline = AsyncNLPPipeline(classifier, searcher, confidence_threshold=0.6)

# Process query (async)
result = await pipeline.run("আমার NID কার্ড হারিয়ে গেছে", top_k=5, tenant_id="default")

# result.classification contains top-3 predictions
# result.search_results contains semantic search results
# result.strategy indicates "single_cluster" or "merged_fallback"
```

---

## Testing

**File:** `tests/unit/test_models_new.py`

Contains tests for model base classes and registry. The classifier follows the same `BaseModel` interface.

**Running Tests:**
```bash
pytest tests/unit/test_models_new.py
pytest tests/integration/  # For end-to-end pipeline tests
```

---

## Performance Metrics

Current model performance (from `models/classification/validation_metrics.json`):

- **Balanced Accuracy:** ~0.91
- **Architecture:** Logistic Regression over E5-large embeddings
- **Classes:** 48 support clusters
- **Training Data:** ~3,200 Bengali dialogue utterances

---

## Training a New Classifier

### Prerequisites

1. Ensure datasets are prepared:
   ```bash
   python pipelines/data/prepare_datasets.py
   ```

2. Configure hyperparameters in `config/classification.json`

### Training Steps

```bash
# Full training with hyperparameter search
python pipelines/training/train_classifier.py

# The script will:
# 1. Load E5 model
# 2. Generate embeddings for train/val sets
# 3. Train classifier (with optional grid search)
# 4. Evaluate on validation set
# 5. Save artifacts to models/classification/
```

### Advanced Options

Enable hyperparameter search in `config/classification.json`:
```json
{
  "hyperparameter_search": {
    "enabled": true,
    "C_values": [0.1, 0.5, 1.0, 2.0, 5.0],
    "cv_folds": 5
  }
}
```

Enable probability calibration:
```json
{
  "calibration": {
    "enabled": true,
    "method": "sigmoid"  // or "isotonic"
  }
}
```

---

## Fractional Query Classifier

**File:** `src/application/context/fraction_classifier.py`

A secondary classifier that detects context-dependent "fractional" queries:

- Trained on `datasets/raw/fractional_queries.txt`
- Used for dialogue context augmentation
- Binary classification (fractional vs complete)
- Training script: `pipelines/training/train_fractional_classifier.py`

---

## Related Components

### Semantic Search

**File:** `src/application/models/search.py`

Works in tandem with classifier:
- Uses classification confidence to select search strategy
- Searches in predicted cluster or merged index

### Policy Engine

**File:** `src/application/policy/handlers.py`

Uses classification results for routing:
- High confidence → direct FAQ answer
- Low confidence → clarification or escalation
- Cluster determines form triggers

---

## Troubleshooting

### Model Not Found

```python
FileNotFoundError: models/classification/model.pkl not found
```

**Solution:** Train the classifier first:
```bash
python pipelines/training/train_classifier.py
```

### Low Accuracy

- Check training data quality in `datasets/raw/questions_with_clusters.csv`
- Enable hyperparameter search in config
- Try probability calibration
- Increase training data size

### Slow Predictions

- Ensure E5 embeddings are cached
- Verify classifier runs in thread pool (async)
- Check circuit breaker isn't tripping

---

## Architecture Summary

```
User Query
    ↓
[AsyncNLPPipeline]
    ↓
[Thread Pool] → [E5 Embeddings] → [ClassifierModel.predict()]
    ↓                                       ↓
    ├─ High Confidence (≥0.6) → Search in Cluster Index
    └─ Low Confidence (<0.6) → Search in Merged Index
```

---

## References

- **Main README:** `/README.md` - Project overview
- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.txt` - Full system design
- **Config Schema:** `src/application/config/schemas.py` - Configuration types
- **Model Base Class:** `src/application/models/base.py` - Model interface
