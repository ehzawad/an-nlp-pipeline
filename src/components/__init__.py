"""
Composeable NLP Components

This module provides standalone, independently runnable NLP components
that can be composed together to build complex dialogue systems.
"""

from src.components.base import (
    Component,
    ComponentInput,
    ComponentOutput,
    ComponentConfig,
)

__all__ = [
    "Component",
    "ComponentInput",
    "ComponentOutput",
    "ComponentConfig",
]
