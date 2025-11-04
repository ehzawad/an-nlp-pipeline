"""
Base interfaces for composeable NLP components.

All components implement the Component protocol and can be:
- Run independently via CLI
- Composed together in pipelines
- Tested in isolation
- Configured via standard interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ComponentInput:
    """Base class for component inputs with standard metadata"""
    text: str
    language: str = "bn"  # Bengali by default
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    tenant_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for CLI/API usage"""
        return json.dumps({
            "text": self.text,
            "language": self.language,
            "context": self.context,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "ComponentInput":
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class ComponentOutput:
    """Base class for component outputs with standard metadata"""
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    component_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for CLI/API usage"""
        return json.dumps({
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "component_name": self.component_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ComponentOutput":
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class ComponentConfig:
    """Base configuration for components"""
    enabled: bool = True
    config_path: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "cpu"
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "config_path": self.config_path,
            "model_path": self.model_path,
            "device": self.device,
            "extra_config": self.extra_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentConfig":
        """Create from dictionary"""
        return cls(**data)


TInput = TypeVar("TInput", bound=ComponentInput)
TOutput = TypeVar("TOutput", bound=ComponentOutput)
TConfig = TypeVar("TConfig", bound=ComponentConfig)


class Component(ABC, Generic[TInput, TOutput, TConfig]):
    """
    Base interface for all NLP components.

    Each component must:
    1. Accept standardized input via process()
    2. Return standardized output
    3. Be configurable via ComponentConfig
    4. Support both sync and async execution
    5. Be independently runnable
    """

    def __init__(self, config: TConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the component (load models, caches, etc.)
        Called once before first use.
        """
        pass

    @abstractmethod
    async def process(self, input_data: TInput) -> TOutput:
        """
        Process input and return output.
        This is the main entry point for the component.
        """
        pass

    async def process_batch(self, inputs: list[TInput]) -> list[TOutput]:
        """
        Process multiple inputs in batch.
        Default implementation processes sequentially.
        Override for better batch performance.
        """
        results = []
        for input_data in inputs:
            result = await self.process(input_data)
            results.append(result)
        return results

    @abstractmethod
    def get_name(self) -> str:
        """Return component name"""
        pass

    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the component.
        Returns status and diagnostics.
        """
        return {
            "name": self.get_name(),
            "initialized": self._initialized,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "status": "healthy" if self._initialized else "not_initialized",
        }

    async def shutdown(self) -> None:
        """
        Cleanup resources (optional).
        Called when component is no longer needed.
        """
        self._initialized = False


class ComponentPipeline:
    """
    Pipeline for composing multiple components.
    Components are executed in sequence, with output of one feeding into next.
    """

    def __init__(self, components: list[Component]):
        self.components = components

    async def initialize_all(self) -> None:
        """Initialize all components in the pipeline"""
        for component in self.components:
            if not component.is_initialized():
                await component.initialize()

    async def process(self, input_data: ComponentInput) -> list[ComponentOutput]:
        """
        Process input through all components.
        Returns list of outputs from each component.
        """
        outputs = []
        current_input = input_data

        for component in self.components:
            output = await component.process(current_input)
            outputs.append(output)

            # Update context with previous component's output
            if output.success and isinstance(current_input, ComponentInput):
                current_input.context[component.get_name()] = output.data

        return outputs

    async def shutdown_all(self) -> None:
        """Shutdown all components"""
        for component in self.components:
            await component.shutdown()
