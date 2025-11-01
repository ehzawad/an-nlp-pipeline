"""Integration tests for tenant context isolation."""

import asyncio

import pytest

from src.shared.tenant_context import TenantConfig, TenantContextFactory, TenantContext


@pytest.mark.asyncio
async def test_tenant_services_isolated(tmp_path):
    factory = TenantContextFactory()
    factory.register_tenant(TenantConfig(tenant_id="alpha", tenant_name="Alpha"))
    factory.register_tenant(TenantConfig(tenant_id="beta", tenant_name="Beta"))

    ctx_alpha = await factory.create_context("alpha")
    ctx_beta = await factory.create_context("beta")

    bus_alpha = await ctx_alpha.get_event_bus()
    bus_beta = await ctx_beta.get_event_bus()

    assert ctx_alpha.tenant_id == "alpha"
    assert ctx_beta.tenant_id == "beta"
    assert bus_alpha is not bus_beta


@pytest.mark.asyncio
async def test_lazy_loading_session_manager(monkeypatch):
    factory = TenantContextFactory()
    factory.register_tenant(TenantConfig(tenant_id="lazy", tenant_name="Lazy"))
    ctx = await factory.create_context("lazy")

    created_instances = []

    async def fake_get_session_manager(self):
        if not hasattr(self, "_fake_manager"):
            self._fake_manager = object()
            created_instances.append(self._fake_manager)
        return self._fake_manager

    monkeypatch.setattr(TenantContext, "get_session_manager", fake_get_session_manager, raising=False)

    manager_one = await ctx.get_session_manager()
    manager_two = await ctx.get_session_manager()

    assert manager_one is manager_two
    assert len(created_instances) == 1


@pytest.mark.asyncio
async def test_tenant_cleanup_releases_resources():
    factory = TenantContextFactory()
    factory.register_tenant(TenantConfig(tenant_id="cleanup", tenant_name="Cleanup"))
    ctx = await factory.create_context("cleanup")

    closed = asyncio.Event()
    unloaded = asyncio.Event()

    class FakeRedis:
        async def close(self):
            closed.set()

    class FakeRegistry:
        async def unload_all(self):
            unloaded.set()

    ctx._redis = FakeRedis()
    ctx._model_registry = FakeRegistry()

    await ctx.cleanup()

    assert closed.is_set()
    assert unloaded.is_set()
