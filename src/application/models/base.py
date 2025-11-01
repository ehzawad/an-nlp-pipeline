"""Base model interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel as PydanticBase, Field


class ModelConfig(PydanticBase):
    """Configuration for a model."""
    name: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type")
    implementation: str = Field(..., description="Implementation class path")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model-specific config")

    class Config:
        extra = "allow"


class BaseModel(ABC):
    """Base class for all models in the system."""

    def __init__(self, config: ModelConfig):
        """Initialize model with configuration."""
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model. Must be called before use."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        pass

    def unload(self) -> None:
        """Unload model and free resources."""
        self._loaded = False

    def reload(self) -> None:
        """Reload the model."""
        self.unload()
        self.load()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', loaded={self._loaded})"

