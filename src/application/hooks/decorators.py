"""Hook decorators for easy registration."""

from functools import wraps
from typing import Callable, Optional
from .manager import HookManager, HookPoint


def hook(
    point: HookPoint | str,
    priority: int = 100,
    name: Optional[str] = None
):
    """
    Decorator to register a function as a hook.
    
    Usage:
        @hook(HookPoint.BEFORE_NLP, priority=50)
        async def my_hook(context: HookContext):
            # Hook logic here
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Store hook metadata on function
        func._hook_point = point
        func._hook_priority = priority
        func._hook_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def auto_register_hooks(hook_manager: HookManager, module):
    """
    Auto-register all hooks in a module.
    
    Usage:
        from . import builtins
        auto_register_hooks(hook_manager, builtins.logging_hook)
    """
    import inspect
    
    for name, obj in inspect.getmembers(module):
        if hasattr(obj, '_hook_point'):
            hook_manager.register(
                point=obj._hook_point,
                handler=obj,
                name=obj._hook_name,
                priority=obj._hook_priority
            )

