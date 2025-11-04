"""
Standalone Classification Component implementation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import time

from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

# Reuse existing models
from src.application.models.classifiers import ClassifierModel
from src.application.models.embeddings import E5EmbeddingModel
from src.application.models.base import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig(ComponentConfig):
    """Configuration for Classification component"""
    model_path: str = "models/classification/model.pkl"
    label_encoder_path: str = "models/classification/label_encoder.pkl"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_cache_dir: str = "./models/embeddings/e5_cache"
    top_k: int = 3
    confidence_threshold: float = 0.6


@dataclass
class ClassificationInput(ComponentInput):
    """Input for Classification component"""
    top_k: Optional[int] = None  # Override default top_k if needed


@dataclass
class PredictionResult:
    """Single classification prediction"""
    cluster: str
    confidence: float


@dataclass
class ClassificationOutput(ComponentOutput):
    """Output from Classification component"""
    predictions: List[PredictionResult] = field(default_factory=list)
    top_cluster: str = ""
    top_confidence: float = 0.0

    def __post_init__(self):
        # Ensure data contains predictions
        if self.data is None or not isinstance(self.data, dict):
            self.data = {
                "predictions": [
                    {"cluster": p.cluster, "confidence": p.confidence}
                    for p in self.predictions
                ],
                "top_cluster": self.top_cluster,
                "top_confidence": self.top_confidence,
            }


class ClassificationComponent(Component[ClassificationInput, ClassificationOutput, ClassificationConfig]):
    """
    Standalone Classification Component.

    Features:
    - Intent classification using E5 embeddings + Logistic Regression
    - Top-K predictions with confidence scores
    - Configurable confidence thresholds
    - Can run independently or in pipeline

    Example usage:
        config = ClassificationConfig(device="cpu")
        classifier = ClassificationComponent(config)
        await classifier.initialize()

        input_data = ClassificationInput(text="আমার NID স্ট্যাটাস কি?")
        output = await classifier.process(input_data)
        print(output.top_cluster, output.top_confidence)
    """

    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.embedding_model: Optional[E5EmbeddingModel] = None
        self.classifier_model: Optional[ClassifierModel] = None

    async def initialize(self) -> None:
        """Initialize embedding model and classifier"""
        if self._initialized:
            logger.info("Classification component already initialized")
            return

        logger.info("Initializing Classification component...")
        start_time = time.time()

        # Initialize embedding model
        embedding_config = ModelConfig(config={
            "name": self.config.embedding_model,
            "cache_dir": self.config.embedding_cache_dir,
            "device": self.config.device,
        })
        self.embedding_model = E5EmbeddingModel(embedding_config)
        self.embedding_model.load()

        # Initialize classifier model
        classifier_config = ModelConfig(config={
            "model_path": self.config.model_path,
            "label_encoder_path": self.config.label_encoder_path,
        })
        self.classifier_model = ClassifierModel(classifier_config, self.embedding_model)
        self.classifier_model.load()

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Classification component initialized in {elapsed:.2f}ms")

    async def process(self, input_data: ClassificationInput) -> ClassificationOutput:
        """Classify text into intent clusters"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Determine top_k
            top_k = input_data.top_k or self.config.top_k

            # Run classification (CPU-bound, but in async context)
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=2)
            loop = asyncio.get_event_loop()

            predictions_raw = await loop.run_in_executor(
                executor,
                self.classifier_model.predict,
                input_data.text,
                top_k
            )

            # Convert to typed predictions
            predictions = [
                PredictionResult(
                    cluster=pred["cluster"],
                    confidence=pred["confidence"]
                )
                for pred in predictions_raw
            ]

            top_cluster = predictions[0].cluster if predictions else ""
            top_confidence = predictions[0].confidence if predictions else 0.0

            processing_time_ms = (time.time() - start_time) * 1000

            return ClassificationOutput(
                success=True,
                data={
                    "predictions": [
                        {"cluster": p.cluster, "confidence": p.confidence}
                        for p in predictions
                    ],
                    "top_cluster": top_cluster,
                    "top_confidence": top_confidence,
                },
                predictions=predictions,
                top_cluster=top_cluster,
                top_confidence=top_confidence,
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

        except Exception as e:
            logger.error(f"Classification processing failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return ClassificationOutput(
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

    def get_name(self) -> str:
        """Return component name"""
        return "classification"

    async def health_check(self) -> Dict[str, Any]:
        """Health check for Classification component"""
        base_health = await super().health_check()

        classification_specific = {
            "embedding_model_loaded": self.embedding_model.is_loaded() if self.embedding_model else False,
            "classifier_loaded": self.classifier_model.is_loaded() if self.classifier_model else False,
        }

        return {**base_health, **classification_specific}


# CLI entry point
async def main():
    """CLI entry point for standalone Classification component"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Standalone Classification Component")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    parser.add_argument("--model-path", type=str, default="models/classification/model.pkl")
    parser.add_argument("--label-encoder-path", type=str, default="models/classification/label_encoder.pkl")
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    # Create config
    config = ClassificationConfig(
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        embedding_model=args.embedding_model,
        device=args.device,
        top_k=args.top_k,
    )

    # Initialize component
    component = ClassificationComponent(config)
    await component.initialize()

    # Prepare input
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_data = ClassificationInput(**input_json)
    elif args.text:
        input_data = ClassificationInput(text=args.text)
    else:
        print("Error: Must provide --text or --input-file", file=sys.stderr)
        sys.exit(1)

    # Process
    output = await component.process(input_data)

    # Write output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output.to_json())
    else:
        print(output.to_json())

    # Exit with appropriate code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
