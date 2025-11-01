"""Pydantic schemas for type-safe configs."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str
    device: str = "cpu"
    cache_dir: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: str = "INFO"
    file: Optional[str] = None


class MainConfig(BaseModel):
    """Main configuration schema."""
    project_name: str
    python_version: str
    paths: Dict[str, str]
    models: Dict[str, ModelConfig]
    logging: LoggingConfig


class ClassificationConfig(BaseModel):
    """Classification configuration schema."""
    model_path: str
    label_encoder_path: str
    confidence_threshold: float = 0.6


class SemanticSearchConfig(BaseModel):
    """Semantic search configuration schema."""
    indices_dir: str
    metadata_path: str
    default_top_k: int = 5
    auto_build_on_missing: bool = False


class FeatureToggles(BaseModel):
    """Feature toggles schema."""
    enable_forms: bool = True
    enable_context_augmentation: bool = True
    enable_summarization: bool = True


class SummarizationConfig(BaseModel):
    """Summarization configuration schema."""
    min_length_to_summarize: int = 100
    max_summary_length: int = 150
    implementation: str = "passthrough"


class ContextAugmentationConfig(BaseModel):
    """Context augmentation configuration schema."""
    context_window_turns: int = 1
    min_confidence: float = 0.7
    mappings_file: str
    classifier_path: str


class RedisConfig(BaseModel):
    """Redis configuration schema."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class SessionConfig(BaseModel):
    """Session configuration schema."""
    ttl_seconds: int = 1800
    max_history_turns: int = 10
    redis_prefix: str = "dialogue:session:"


class PolicyConfig(BaseModel):
    """Policy configuration schema."""
    confidence_threshold: float = 0.7
    clarification_threshold: float = 0.5


class FormsConfig(BaseModel):
    """Forms configuration schema."""
    max_interruptions: int = 3
    cluster_to_form_mapping: Dict[str, str]


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration schema."""
    max_errors_before_escalation: int = 3
    enable_graceful_degradation: bool = True


class DialogueConfig(BaseModel):
    """Dialogue configuration schema."""
    features: FeatureToggles
    summarization: SummarizationConfig
    context_augmentation: ContextAugmentationConfig
    redis: RedisConfig
    session: SessionConfig
    policy: PolicyConfig
    forms: FormsConfig
    error_handling: ErrorHandlingConfig
    logging: LoggingConfig

