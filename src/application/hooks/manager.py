"""Hook manager for registering and executing hooks."""

from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import inspect
import logging

logger = logging.getLogger(__name__)


class HookPoint(str, Enum):
    """Predefined hook points in the system."""
    BEFORE_NLP = "before_nlp"
    AFTER_NLP = "after_nlp"
    BEFORE_POLICY = "before_policy"
    AFTER_POLICY = "after_policy"
    BEFORE_RESPONSE = "before_response"
    AFTER_RESPONSE = "after_response"
    BEFORE_FORM_EXECUTE = "before_form_execute"
    AFTER_FORM_EXECUTE = "after_form_execute"
    ON_ERROR = "on_error"


@dataclass
class HookContext:
    """Context passed to hooks."""
    query: str
    session_id: str
    nlp_result: Optional[Dict[str, Any]] = None
    policy_decision: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a value in metadata."""
        self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata."""
        return self.metadata.get(key, default)


@dataclass
class Hook:
    """A registered hook."""
    name: str
    point: HookPoint
    handler: Callable
    priority: int = 100
    enabled: bool = True

    def __repr__(self) -> str:
        return f"Hook(name='{self.name}', point={self.point.value}, priority={self.priority})"


class HookManager:
    """Central manager for the hook system."""

    def __init__(self):
        """Initialize empty hook registry."""
        self._hooks: Dict[HookPoint, List[Hook]] = {point: [] for point in HookPoint}

    def register(
        self,
        point: HookPoint | str,
        handler: Callable,
        name: Optional[str] = None,
        priority: int = 100
    ) -> Hook:
        """Register a hook."""
        # Convert string to enum
        if isinstance(point, str):
            point = HookPoint(point)

        # Generate name if not provided
        if name is None:
            name = handler.__name__

        # Create hook
        hook_instance = Hook(
            name=name,
            point=point,
            handler=handler,
            priority=priority
        )

        # Add to registry
        self._hooks[point].append(hook_instance)

        # Sort by priority
        self._hooks[point].sort(key=lambda h: h.priority)

        logger.info(f"Registered hook: {name} @ {point.value} (priority={priority})")

        return hook_instance

    def unregister(self, point: HookPoint | str, name: str) -> bool:
        """Unregister a hook by name."""
        if isinstance(point, str):
            point = HookPoint(point)

        hooks = self._hooks[point]
        for i, hook in enumerate(hooks):
            if hook.name == name:
                hooks.pop(i)
                return True
        return False

    def disable(self, point: HookPoint | str, name: str) -> bool:
        """Disable a hook without removing it."""
        if isinstance(point, str):
            point = HookPoint(point)

        for hook in self._hooks[point]:
            if hook.name == name:
                hook.enabled = False
                return True
        return False

    def enable(self, point: HookPoint | str, name: str) -> bool:
        """Enable a previously disabled hook."""
        if isinstance(point, str):
            point = HookPoint(point)

        for hook in self._hooks[point]:
            if hook.name == name:
                hook.enabled = True
                return True
        return False

    async def execute(self, point: HookPoint | str, context: HookContext) -> HookContext:
        """Execute all hooks at a given point."""
        if isinstance(point, str):
            point = HookPoint(point)

        hooks = self._hooks[point]

        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                # Check if handler is async
                if inspect.iscoroutinefunction(hook.handler):
                    await hook.handler(context)
                else:
                    hook.handler(context)

            except Exception as e:
                logger.error(f"Hook '{hook.name}' failed: {e}")
                context.set(f"hook_error_{hook.name}", str(e))

        return context

    def get_hooks(self, point: Optional[HookPoint | str] = None) -> List[Hook]:
        """Get registered hooks."""
        if point is None:
            all_hooks = []
            for hooks in self._hooks.values():
                all_hooks.extend(hooks)
            return all_hooks
        else:
            if isinstance(point, str):
                point = HookPoint(point)
            return self._hooks[point]

    def clear(self, point: Optional[HookPoint | str] = None) -> None:
        """Clear hooks."""
        if point is None:
            for p in HookPoint:
                self._hooks[p] = []
        else:
            if isinstance(point, str):
                point = HookPoint(point)
            self._hooks[point] = []

    def __repr__(self) -> str:
        total = sum(len(hooks) for hooks in self._hooks.values())
        return f"HookManager(hooks={total})"

