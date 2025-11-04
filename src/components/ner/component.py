"""
Standalone NER Component implementation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import time

from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

# Reuse existing NER infrastructure
from src.application.ner.ner_extractor import NERExtractor
from src.application.ner.entity_patterns import EntityPatterns
from src.application.ner.entity_mapper import EntityMapper

logger = logging.getLogger(__name__)


@dataclass
class NERConfig(ComponentConfig):
    """Configuration for NER component"""
    model_name: str = "xlm-roberta-large-finetuned-conll03-english"
    confidence_threshold: float = 0.7
    enable_regex_fallback: bool = True
    enable_transformer: bool = True


@dataclass
class NERInput(ComponentInput):
    """Input for NER component"""
    form_slots: Optional[List[str]] = None  # Optional form slots for mapping
    entity_types: Optional[List[str]] = None  # Specific entity types to extract


@dataclass
class NEROutput(ComponentOutput):
    """Output from NER component"""
    entities: Dict[str, str] = field(default_factory=dict)
    raw_entities: List[tuple] = field(default_factory=list)
    extraction_method: str = "none"  # "hybrid", "bert", "regex", "none"

    def __post_init__(self):
        # Ensure data contains the entities
        if self.data is None or not isinstance(self.data, dict):
            self.data = {
                "entities": self.entities,
                "raw_entities": self.raw_entities,
                "extraction_method": self.extraction_method,
            }


class NERComponent(Component[NERInput, NEROutput, NERConfig]):
    """
    Standalone NER Component.

    Features:
    - Hybrid extraction (transformer + regex)
    - Configurable confidence thresholds
    - Entity-to-slot mapping
    - Can run independently or in pipeline

    Example usage:
        config = NERConfig(device="cpu")
        ner = NERComponent(config)
        await ner.initialize()

        input_data = NERInput(
            text="আমার নাম জন ডো এবং আমার NID 12345678901234567",
            form_slots=["name", "nid_number"]
        )

        output = await ner.process(input_data)
        print(output.entities)  # {"name": "জন ডো", "nid_number": "12345678901234567"}
    """

    def __init__(self, config: NERConfig):
        super().__init__(config)
        self.extractor: Optional[NERExtractor] = None
        self.pattern_matcher: Optional[EntityPatterns] = None
        self.entity_mapper: Optional[EntityMapper] = None

    async def initialize(self) -> None:
        """Initialize NER models and patterns"""
        if self._initialized:
            logger.info("NER component already initialized")
            return

        logger.info("Initializing NER component...")
        start_time = time.time()

        # Initialize pattern matcher and entity mapper (always available)
        self.pattern_matcher = EntityPatterns()
        self.entity_mapper = EntityMapper()

        # Initialize transformer-based extractor if enabled
        if self.config.enable_transformer:
            self.extractor = NERExtractor(
                model_name=self.config.model_name,
                cache_dir=self.config.model_path,
                device=self.config.device,
                confidence_threshold=self.config.confidence_threshold,
                enable_regex_fallback=self.config.enable_regex_fallback,
            )
            # Load the model
            self.extractor.load()
        else:
            logger.info("Transformer NER disabled, using regex-only mode")

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"NER component initialized in {elapsed:.2f}ms")

    async def process(self, input_data: NERInput) -> NEROutput:
        """Process text and extract entities"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # If transformer is enabled, use the full extractor
            if self.extractor:
                result = await self.extractor.extract(
                    text=input_data.text,
                    form_slots=input_data.form_slots,
                )
                entities = result["entities"]
                raw_entities = result["raw_entities"]
                method = result["method"]
            else:
                # Regex-only mode
                entities = await self._extract_with_regex_only(
                    text=input_data.text,
                    form_slots=input_data.form_slots,
                )
                raw_entities = list(entities.items())
                method = "regex"

            processing_time_ms = (time.time() - start_time) * 1000

            return NEROutput(
                success=True,
                data={
                    "entities": entities,
                    "raw_entities": raw_entities,
                    "extraction_method": method,
                },
                entities=entities,
                raw_entities=raw_entities,
                extraction_method=method,
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

        except Exception as e:
            logger.error(f"NER processing failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return NEROutput(
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

    async def _extract_with_regex_only(
        self,
        text: str,
        form_slots: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Extract entities using regex patterns only"""
        regex_results = self.pattern_matcher.extract_all(text)
        entities = self.pattern_matcher.prioritize_entities(regex_results)

        # Map to form slots if provided
        if form_slots and self.entity_mapper:
            entities = self.entity_mapper.map_entities_to_slots(entities, form_slots)

        return entities

    async def process_batch(self, inputs: List[NERInput]) -> List[NEROutput]:
        """Process multiple inputs in batch"""
        # For NER, we can process in parallel since each is independent
        import asyncio
        tasks = [self.process(input_data) for input_data in inputs]
        return await asyncio.gather(*tasks)

    def get_name(self) -> str:
        """Return component name"""
        return "ner"

    async def health_check(self) -> Dict[str, Any]:
        """Health check for NER component"""
        base_health = await super().health_check()

        ner_specific = {
            "transformer_enabled": self.config.enable_transformer,
            "regex_enabled": self.config.enable_regex_fallback,
            "model_loaded": self.extractor.is_loaded() if self.extractor else False,
        }

        return {**base_health, **ner_specific}


# CLI entry point
async def main():
    """CLI entry point for standalone NER component"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Standalone NER Component")
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    parser.add_argument("--model", type=str, default="xlm-roberta-large-finetuned-conll03-english")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--regex-only", action="store_true", help="Use regex-only mode")
    parser.add_argument("--slots", type=str, help="Comma-separated form slots")

    args = parser.parse_args()

    # Create config
    config = NERConfig(
        model_name=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        enable_transformer=not args.regex_only,
        enable_regex_fallback=True,
    )

    # Initialize component
    component = NERComponent(config)
    await component.initialize()

    # Prepare input
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_data = NERInput(**input_json)
    elif args.text:
        form_slots = args.slots.split(",") if args.slots else None
        input_data = NERInput(text=args.text, form_slots=form_slots)
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
