"""Config loader with override support."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Project root (go up from src/application/config/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ConfigLoader:
    """Load JSON configs with local override support."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        if config_dir is None:
            base_dir = PROJECT_ROOT / "config"
        else:
            base_dir = Path(config_dir)
            if not base_dir.is_absolute():
                base_dir = PROJECT_ROOT / base_dir

        self.project_root = PROJECT_ROOT
        self.config_dir = base_dir.resolve()
        self._cache = {}

    def load(self, config_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load config with optional overrides.
        
        Args:
            config_name: Name of config file (e.g., 'main', 'classification')
            overrides: Dict of key-value pairs to override
            
        Returns:
            Merged config dict
        """
        config_path = self.config_dir / f"{config_name}.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path in self._cache:
            config = self._cache[config_path].copy()
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._cache[config_path] = config.copy()

        # Apply overrides
        if overrides:
            config = self._deep_merge(config, overrides)
            logger.info(f"Applied overrides to {config_name}: {list(overrides.keys())}")

        return config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a repository-relative path to an absolute path.
        
        Args:
            path: Relative or absolute path
            
        Returns:
            Absolute Path within the project
        """
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def clear_cache(self):
        """Clear cached configs."""
        self._cache.clear()

