"""Tests for tenant context and multi-tenancy support."""

import pytest
import asyncio
from pathlib import Path

from src.shared.tenant_context import (
    TenantConfig, TenantContext, TenantContextFactory
)


class TestTenantConfig:
    """Tests for TenantConfig."""

    def test_tenant_config_initialization(self):
        """Test tenant config creates with defaults."""
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test Tenant"
        )

        assert config.tenant_id == "test_tenant"
        assert config.tenant_name == "Test Tenant"
        assert config.enable_forms is True
        assert config.enable_context_augmentation is True
        assert config.enable_ner is True
        assert config.confidence_threshold == 0.7
        assert config.clarification_threshold == 0.5
        assert config.max_sessions_per_user == 5
        assert config.max_turns_per_session == 100
        assert config.session_ttl_seconds == 1800
        assert config.model_variant == "standard"
        assert config.language == "bn"
        # Note: redis_db field exists but not used (in-memory storage)

    def test_tenant_config_cache_prefix(self):
        """Test cache prefix is auto-generated from tenant_id."""
        config = TenantConfig(
            tenant_id="tenant_123",
            tenant_name="Test"
        )

        assert config.cache_prefix == "tenant:tenant_123"

    def test_tenant_config_custom_settings(self):
        """Test tenant config with custom settings."""
        config = TenantConfig(
            tenant_id="premium_tenant",
            tenant_name="Premium",
            enable_forms=False,
            enable_summarization=True,
            confidence_threshold=0.8,
            max_sessions_per_user=10,
            model_variant="premium",
            language="en"
        )

        assert config.enable_forms is False
        assert config.enable_summarization is True
        assert config.confidence_threshold == 0.8
        assert config.max_sessions_per_user == 10
        assert config.model_variant == "premium"
        assert config.language == "en"


class TestTenantContext:
    """Tests for TenantContext."""

    @pytest.mark.asyncio
    async def test_tenant_context_initialization(self):
        """Test tenant context initializes correctly."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(
            tenant_id="test",
            config=config,
            correlation_id="abc-123"
        )

        assert context.tenant_id == "test"
        assert context.config == config
        assert context.correlation_id == "abc-123"
        assert context._event_store is None
        assert context._event_bus is None

    @pytest.mark.asyncio
    async def test_get_event_store(self):
        """Test lazy initialization of event store."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        # First call should create event store
        event_store1 = await context.get_event_store()
        assert event_store1 is not None
        assert context._event_store is event_store1

        # Second call should return same instance
        event_store2 = await context.get_event_store()
        assert event_store2 is event_store1

    @pytest.mark.asyncio
    async def test_get_event_bus(self):
        """Test lazy initialization of event bus."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        # First call should create event bus
        event_bus1 = await context.get_event_bus()
        assert event_bus1 is not None
        assert context._event_bus is event_bus1

        # Second call should return same instance
        event_bus2 = await context.get_event_bus()
        assert event_bus2 is event_bus1

    @pytest.mark.asyncio
    async def test_get_session_manager(self):
        """Test lazy initialization of session manager."""
        from src.application.session import InMemorySessionStore
        
        factory = TenantContextFactory()
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        factory.register_tenant(config)
        
        # Register session store for the tenant
        store = InMemorySessionStore(ttl_seconds=1800)
        factory.register_session_store("test", store)
        
        context = await factory.create_context("test")

        session_manager = await context.get_session_manager()
        assert session_manager is not None
        assert context._session_manager is session_manager

    @pytest.mark.asyncio
    async def test_get_hook_manager(self):
        """Test lazy initialization of hook manager."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        hook_manager = await context.get_hook_manager()
        assert hook_manager is not None
        assert context._hook_manager is hook_manager

    @pytest.mark.asyncio
    async def test_get_ner_extractor(self):
        """Test lazy initialization of NER extractor."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        ner_extractor = await context.get_ner_extractor()
        assert ner_extractor is not None
        assert context._ner_extractor is ner_extractor

    @pytest.mark.asyncio
    async def test_get_summarizer(self):
        """Test lazy initialization of summarizer."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        summarizer = await context.get_summarizer()
        assert summarizer is not None
        assert context._summarizer is summarizer

    @pytest.mark.asyncio
    async def test_get_form_registry(self):
        """Test lazy initialization of form registry."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        form_registry = await context.get_form_registry()
        assert form_registry is not None
        assert context._form_registry is form_registry

        # Verify forms are registered
        from src.application.forms.examples import NIDStatusCheckForm
        assert form_registry.get(NIDStatusCheckForm().form_name) is not None
        assert len(form_registry) == 4  # All 4 example forms registered

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test that concurrent access to lazy-loaded services is thread-safe."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        # Try to get event store from multiple coroutines concurrently
        results = await asyncio.gather(
            context.get_event_store(),
            context.get_event_store(),
            context.get_event_store()
        )

        # All should return the same instance
        assert results[0] is results[1]
        assert results[1] is results[2]

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method."""
        config = TenantConfig(tenant_id="test", tenant_name="Test")
        context = TenantContext(tenant_id="test", config=config)

        # Initialize some services
        await context.get_event_store()
        await context.get_event_bus()

        # Cleanup should not raise errors
        await context.cleanup()


class TestTenantContextFactory:
    """Tests for TenantContextFactory."""

    def test_factory_initialization(self):
        """Test factory initializes with empty configs."""
        factory = TenantContextFactory()
        assert len(factory._tenant_configs) == 0

    def test_register_tenant(self):
        """Test registering tenant configurations."""
        factory = TenantContextFactory()

        config1 = TenantConfig(tenant_id="tenant1", tenant_name="Tenant 1")
        config2 = TenantConfig(tenant_id="tenant2", tenant_name="Tenant 2")

        factory.register_tenant(config1)
        factory.register_tenant(config2)

        assert len(factory._tenant_configs) == 2
        assert factory._tenant_configs["tenant1"] == config1
        assert factory._tenant_configs["tenant2"] == config2

    @pytest.mark.asyncio
    async def test_create_context(self):
        """Test creating tenant context."""
        factory = TenantContextFactory()

        config = TenantConfig(tenant_id="test", tenant_name="Test")
        factory.register_tenant(config)

        context = await factory.create_context("test", correlation_id="xyz-456")

        assert context.tenant_id == "test"
        assert context.config == config
        assert context.correlation_id == "xyz-456"

    @pytest.mark.asyncio
    async def test_create_context_unknown_tenant(self):
        """Test creating context for unknown tenant raises error."""
        factory = TenantContextFactory()

        with pytest.raises(ValueError, match="Unknown tenant: unknown"):
            await factory.create_context("unknown")

    def test_get_tenant_config(self):
        """Test retrieving tenant config."""
        factory = TenantContextFactory()

        config = TenantConfig(tenant_id="test", tenant_name="Test")
        factory.register_tenant(config)

        retrieved = factory.get_tenant_config("test")
        assert retrieved == config

    def test_get_tenant_config_not_found(self):
        """Test retrieving non-existent tenant config returns None."""
        factory = TenantContextFactory()

        retrieved = factory.get_tenant_config("nonexistent")
        assert retrieved is None

    def test_overwrite_tenant_config(self):
        """Test overwriting tenant configuration."""
        factory = TenantContextFactory()

        config1 = TenantConfig(tenant_id="test", tenant_name="Test 1")
        config2 = TenantConfig(tenant_id="test", tenant_name="Test 2")

        factory.register_tenant(config1)
        factory.register_tenant(config2)

        retrieved = factory.get_tenant_config("test")
        assert retrieved.tenant_name == "Test 2"
