"""Tests for the hook management system in src."""

import pytest

from src.application.hooks.manager import HookManager, HookContext, HookPoint


@pytest.mark.asyncio
async def test_register_and_execute_sync_hook(hook_manager_new):
    called = []

    def handler(ctx: HookContext):
        called.append(ctx.query)

    hook_manager_new.register(HookPoint.BEFORE_NLP, handler, name="sync")
    context = HookContext(query="hello", session_id="s1")

    await hook_manager_new.execute(HookPoint.BEFORE_NLP, context)

    assert called == ["hello"]


@pytest.mark.asyncio
async def test_register_async_hook(hook_manager_new):
    called = []

    async def async_handler(ctx: HookContext):
        called.append(ctx.session_id)

    hook_manager_new.register(HookPoint.AFTER_RESPONSE, async_handler, name="async")

    context = HookContext(query="hi", session_id="s2")
    await hook_manager_new.execute(HookPoint.AFTER_RESPONSE, context)

    assert called == ["s2"]


@pytest.mark.asyncio
async def test_priority_order_enforced(hook_manager_new):
    order = []

    hook_manager_new.register(HookPoint.BEFORE_POLICY, lambda ctx: order.append("low"), name="low", priority=200)
    hook_manager_new.register(HookPoint.BEFORE_POLICY, lambda ctx: order.append("high"), name="high", priority=10)
    hook_manager_new.register(HookPoint.BEFORE_POLICY, lambda ctx: order.append("mid"), name="mid", priority=100)

    await hook_manager_new.execute(HookPoint.BEFORE_POLICY, HookContext(query="x", session_id="s3"))

    assert order == ["high", "mid", "low"]


@pytest.mark.asyncio
async def test_disable_and_enable_hook(hook_manager_new):
    called = []

    hook_manager_new.register(HookPoint.BEFORE_RESPONSE, lambda ctx: called.append("first"), name="first")
    hook_manager_new.disable(HookPoint.BEFORE_RESPONSE, "first")

    await hook_manager_new.execute(HookPoint.BEFORE_RESPONSE, HookContext(query="x", session_id="s4"))
    assert called == []

    hook_manager_new.enable(HookPoint.BEFORE_RESPONSE, "first")
    await hook_manager_new.execute(HookPoint.BEFORE_RESPONSE, HookContext(query="x", session_id="s4"))
    assert called == ["first"]


@pytest.mark.asyncio
async def test_unregister_hook(hook_manager_new):
    hook_manager_new.register(HookPoint.AFTER_NLP, lambda ctx: None, name="to_remove")
    removed = hook_manager_new.unregister(HookPoint.AFTER_NLP, "to_remove")

    assert removed is True
    assert hook_manager_new.get_hooks(HookPoint.AFTER_NLP) == []


@pytest.mark.asyncio
async def test_execute_collects_errors(hook_manager_new):
    def bad_handler(ctx: HookContext):
        raise RuntimeError("boom")

    hook_manager_new.register(HookPoint.ON_ERROR, bad_handler, name="boom_hook")

    context = HookContext(query="q", session_id="s5")
    result = await hook_manager_new.execute(HookPoint.ON_ERROR, context)

    assert result.metadata["hook_error_boom_hook"] == "boom"


def test_clear_hooks(hook_manager_new):
    hook_manager_new.register(HookPoint.BEFORE_NLP, lambda ctx: None, name="a")
    hook_manager_new.register(HookPoint.AFTER_NLP, lambda ctx: None, name="b")

    hook_manager_new.clear(HookPoint.BEFORE_NLP)
    assert hook_manager_new.get_hooks(HookPoint.BEFORE_NLP) == []
    assert len(hook_manager_new.get_hooks(HookPoint.AFTER_NLP)) == 1


def test_get_hooks_all(hook_manager_new):
    hook_manager_new.register(HookPoint.BEFORE_POLICY, lambda ctx: None, name="c")
    hook_manager_new.register(HookPoint.BEFORE_RESPONSE, lambda ctx: None, name="d")

    hooks = hook_manager_new.get_hooks()
    names = sorted(hook.name for hook in hooks)

    assert names == ["c", "d"]
