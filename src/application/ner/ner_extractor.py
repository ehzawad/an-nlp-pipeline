"""BERT-based Named Entity Recognition with regex fallback."""

from typing import Dict, List, Optional, Any
import logging
from .entity_patterns import EntityPatterns
from .entity_mapper import EntityMapper

logger = logging.getLogger(__name__)


class NERExtractor:
    """
    Named Entity Recognition extractor.
    
    Combines:
    1. BERT/transformer NER for general entities
    2. Regex patterns for structured IDs
    3. Entity mapper to convert NER labels to slot names
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large-finetuned-conll03-english",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.7,
        enable_regex_fallback: bool = True
    ):
        """Initialize NER extractor."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.enable_regex_fallback = enable_regex_fallback

        # Will be lazily loaded
        self.ner_pipeline = None
        self.tokenizer = None
        self.model = None

        # Initialize pattern matcher and entity mapper
        self.pattern_matcher = EntityPatterns()
        self.entity_mapper = EntityMapper()

        self._loaded = False

    def load(self):
        """Load NER model and tokenizer."""
        if self._loaded:
            logger.info("NER model already loaded")
            return

        logger.info(f"Loading NER model: {self.model_name}")

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            import torch

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Create pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )

            self._loaded = True
            logger.info(f"NER model loaded successfully (device: {self.device})")

        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            if self.enable_regex_fallback:
                logger.warning("Will use regex-only fallback")
                self._loaded = True
            else:
                raise

    def is_loaded(self) -> bool:
        """Check if NER model is loaded."""
        return self._loaded

    async def extract(self, text: str, form_slots: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract entities from text (async-compatible)."""
        if not self._loaded:
            self.load()

        logger.debug(f"Extracting entities from: '{text[:100]}...'")

        # Extract using BERT
        bert_entities = self._extract_with_bert(text) if self.ner_pipeline else {}

        # Extract using regex patterns
        regex_entities = self._extract_with_regex(text) if self.enable_regex_fallback else {}

        # Merge entities (regex takes precedence)
        merged_entities = {**bert_entities, **regex_entities}

        # Map to form slots if provided
        if form_slots:
            mapped_entities = self.entity_mapper.map_entities_to_slots(
                merged_entities,
                form_slots
            )
        else:
            mapped_entities = merged_entities

        method = self._determine_method(bert_entities, regex_entities)

        result = {
            "entities": mapped_entities,
            "raw_entities": list(merged_entities.items()),
            "method": method
        }

        logger.info(
            f"Extracted {len(mapped_entities)} entities via {method}: "
            f"{list(mapped_entities.keys())}"
        )

        return result

    def _extract_with_bert(self, text: str) -> Dict[str, str]:
        """Extract entities using BERT NER model."""
        if not self.ner_pipeline:
            return {}

        try:
            ner_results = self.ner_pipeline(text)
            entities = {}

            for entity in ner_results:
                entity_label = entity['entity_group']
                entity_text = entity['word'].strip()
                confidence = entity['score']

                if confidence < self.confidence_threshold:
                    logger.debug(
                        f"Skipping low confidence entity: {entity_text} "
                        f"({entity_label}, conf={confidence:.2f})"
                    )
                    continue

                entity_type = self.entity_mapper.map_ner_label(entity_label)

                if entity_type:
                    entities[entity_type] = entity_text
                    logger.debug(
                        f"BERT extracted: {entity_type} = '{entity_text}' "
                        f"(conf={confidence:.2f})"
                    )

            return entities

        except Exception as e:
            logger.error(f"BERT NER extraction failed: {e}")
            return {}

    def _extract_with_regex(self, text: str) -> Dict[str, str]:
        """Extract entities using regex patterns."""
        try:
            regex_results = self.pattern_matcher.extract_all(text)
            entities = self.pattern_matcher.prioritize_entities(regex_results)

            if entities:
                logger.debug(f"Regex extracted: {list(entities.keys())}")

            return entities

        except Exception as e:
            logger.error(f"Regex extraction failed: {e}")
            return {}

    def _determine_method(
        self,
        bert_entities: Dict[str, str],
        regex_entities: Dict[str, str]
    ) -> str:
        """Determine which method(s) were used."""
        if bert_entities and regex_entities:
            return "hybrid"
        elif bert_entities:
            return "bert"
        elif regex_entities:
            return "regex"
        else:
            return "none"

    async def extract_for_slot(
        self,
        text: str,
        slot_name: str,
        slot_type: Optional[str] = None
    ) -> Optional[str]:
        """Extract entity for a specific slot (async)."""
        entity_type = self.entity_mapper.get_slot_entity_type(slot_name)

        if entity_type:
            # Try regex first for structured data
            if entity_type in ['nid_number', 'account_number', 'phone', 'email']:
                value = self.pattern_matcher.extract_entity_type(text, entity_type)
                if value:
                    logger.debug(f"Regex extracted '{value}' for slot '{slot_name}'")
                    return value

            # Fall back to full extraction
            result = await self.extract(text, form_slots=[slot_name])
            return result['entities'].get(slot_name)

        return None

