"""Model registry for managing model instances."""

from typing import Dict, Optional
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all models in the system.
    
    Models are loaded on demand and cached for reuse.
    Supports embeddings pointer system for shared model instances.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, BaseModel] = {}

    def register(self, name: str, model: BaseModel) -> None:
        """
        Register a model instance.
        
        Args:
            name: Model identifier (e.g., "e5", "classifier", "searcher")
            model: Model instance
        """
        self._models[name] = model
        logger.info(f"Registered model: {name}")

    def get(self, name: str) -> Optional[BaseModel]:
        """
        Get a registered model by name.
        
        Args:
            name: Model identifier
            
        Returns:
            Model instance or None if not found
        """
        model = self._models.get(name)
        
        # Lazy load if not already loaded
        if model and not model.is_loaded():
            logger.info(f"Lazy loading model: {name}")
            model.load()
        
        return model

    def load_all(self) -> None:
        """Load all registered models."""
        logger.info("Loading all models...")
        
        for name, model in self._models.items():
            if not model.is_loaded():
                logger.info(f"Loading {name}...")
                model.load()
        
        logger.info(f"All models loaded ({len(self._models)} models)")

    def unload_all(self) -> None:
        """Unload all models and free resources."""
        for model in self._models.values():
            if model.is_loaded():
                model.unload()

    def list_models(self) -> Dict[str, bool]:
        """
        List all registered models and their loaded status.
        
        Returns:
            Dict of {model_name: is_loaded}
        """
        return {
            name: model.is_loaded()
            for name, model in self._models.items()
        }

    def __repr__(self) -> str:
        loaded = sum(1 for m in self._models.values() if m.is_loaded())
        total = len(self._models)
        return f"ModelRegistry(models={total}, loaded={loaded})"

