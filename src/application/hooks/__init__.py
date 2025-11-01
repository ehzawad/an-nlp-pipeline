"""Hooks system for dialogue flow interception."""

from .manager import HookManager, HookPoint, HookContext, Hook
from .decorators import hook, auto_register_hooks
from . import builtins

__all__ = [
    "HookManager",
    "HookPoint",
    "HookContext",
    "Hook",
    "hook",
    "auto_register_hooks",
    "builtins",
]

