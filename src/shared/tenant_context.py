"""
Tenant Context - Replaces ALL global singletons with tenant-scoped instances.

Every request gets its own isolated context with:
- Tenant-specific configuration
- Tenant-specific cache
- Tenant-specific models
- Tenant-specific database connection
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import logging

# CRITICAL: Import ML libraries FIRST to avoid SIGSEGV crash
# PyTorch and related libraries must be imported before application code
import torch
import sentence_transformers
import faiss
import numpy as np

from src.infrastructure.event_store import EventStore, EventBus

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Tenant-specific configuration"""
    
    tenant_id: str
    tenant_name: str
    
    # Feature flags per tenant
    enable_forms: bool = True
    enable_context_augmentation: bool = True
    enable_summarization: bool = False
    enable_ner: bool = True
    
    # Tenant-specific thresholds
    confidence_threshold: float = 0.7
    clarification_threshold: float = 0.5
    
    # Resource limits per tenant
    max_sessions_per_user: int = 5
    max_turns_per_session: int = 100
    session_ttl_seconds: int = 1800
    
    # Model configuration per tenant
    model_variant: str = "standard"  # standard, premium, custom
    language: str = "bn"  # Bengali
    
    # Storage configuration
    redis_db: int = 0  # Tenant-specific Redis DB
    cache_prefix: str = ""
    
    def __post_init__(self):
        self.cache_prefix = f"tenant:{self.tenant_id}"


@dataclass
class TenantContext:
    """
    Tenant-scoped context - NO MORE GLOBAL SINGLETONS!
    
    Every component receives context as dependency injection.
    """
    
    tenant_id: str
    config: TenantConfig
    factory: Optional['TenantContextFactory'] = None  # Reference to factory for shared resources
    
    # Tenant-specific services (lazy initialized)
    _event_store: Optional[EventStore] = None
    _event_bus: Optional[EventBus] = None
    _cache: Optional[Any] = None
    _redis: Optional[Any] = None
    _model_registry: Optional[Any] = None
    _session_manager: Optional[Any] = None
    _hook_manager: Optional[Any] = None
    _ner_extractor: Optional[Any] = None
    _context_augmenter: Optional[Any] = None
    _summarizer: Optional[Any] = None
    _nlp_service: Optional[Any] = None
    _policy_engine: Optional[Any] = None
    _form_registry: Optional[Any] = None
    
    # Request tracing
    correlation_id: Optional[str] = None
    
    # Lazy initialization locks
    _init_locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    
    async def get_event_store(self) -> EventStore:
        """Get tenant-specific event store (lazy init)"""
        if self._event_store is None:
            if "event_store" not in self._init_locks:
                self._init_locks["event_store"] = asyncio.Lock()
            
            async with self._init_locks["event_store"]:
                if self._event_store is None:
                    storage_dir = f"./event_store/{self.tenant_id}"
                    self._event_store = EventStore(storage_dir)
                    await self._event_store.load_from_disk()
        
        return self._event_store
    
    async def get_event_bus(self) -> EventBus:
        """Get tenant-specific event bus (lazy init)"""
        if self._event_bus is None:
            if "event_bus" not in self._init_locks:
                self._init_locks["event_bus"] = asyncio.Lock()
            
            async with self._init_locks["event_bus"]:
                if self._event_bus is None:
                    self._event_bus = EventBus()
        
        return self._event_bus
    
    async def get_cache(self) -> Any:
        """Get tenant-specific cache (lazy init)"""
        if self._cache is None:
            # Implementation would create tenant-specific cache instance
            pass
        return self._cache
    
    async def get_redis(self) -> Any:
        """Get tenant-specific Redis connection (lazy init)"""
        if self._redis is None:
            # Implementation would create Redis connection with tenant DB
            pass
        return self._redis
    
    async def get_model_registry(self) -> Any:
        """Get tenant-specific model registry (lazy init)"""
        if self._model_registry is None:
            if "model_registry" not in self._init_locks:
                self._init_locks["model_registry"] = asyncio.Lock()
            
            async with self._init_locks["model_registry"]:
                if self._model_registry is None:
                    from src.application.models import ModelRegistry
                    self._model_registry = ModelRegistry()
        
        return self._model_registry
    
    async def get_session_manager(self) -> Any:
        """Get tenant-specific session manager (lazy init)"""
        if self._session_manager is None:
            if "session_manager" not in self._init_locks:
                self._init_locks["session_manager"] = asyncio.Lock()
            
            async with self._init_locks["session_manager"]:
                if self._session_manager is None:
                    from src.application.session import SessionManager
                    
                    # Get SHARED session store from factory
                    if self.factory is None:
                        raise RuntimeError(f"TenantContext has no factory reference")
                    
                    store = self.factory.get_session_store(self.tenant_id)
                    if store is None:
                        raise RuntimeError(f"No session store registered for tenant {self.tenant_id}")
                    
                    self._session_manager = SessionManager(store)
                    logger.info(f"✅ SessionManager initialized (using shared store)")
        
        return self._session_manager
    
    async def get_hook_manager(self) -> Any:
        """Get tenant-specific hook manager (lazy init)"""
        if self._hook_manager is None:
            if "hook_manager" not in self._init_locks:
                self._init_locks["hook_manager"] = asyncio.Lock()
            
            async with self._init_locks["hook_manager"]:
                if self._hook_manager is None:
                    from src.application.hooks import HookManager
                    from src.application.hooks.builtins import logging_hook, metrics_hook
                    from src.application.hooks.decorators import auto_register_hooks

                    self._hook_manager = HookManager()
                    # Register built-in hooks
                    auto_register_hooks(self._hook_manager, logging_hook)
                    auto_register_hooks(self._hook_manager, metrics_hook)
        
        return self._hook_manager
    
    async def get_ner_extractor(self) -> Any:
        """Get tenant-specific NER extractor (lazy init)"""
        if self._ner_extractor is None:
            if "ner_extractor" not in self._init_locks:
                self._init_locks["ner_extractor"] = asyncio.Lock()
            
            async with self._init_locks["ner_extractor"]:
                if self._ner_extractor is None:
                    from src.application.ner import NERExtractor
                    self._ner_extractor = NERExtractor()
        
        return self._ner_extractor
    
    async def get_context_augmenter(self) -> Any:
        """Get tenant-specific context augmenter (lazy init)"""
        if self._context_augmenter is None:
            if "context_augmenter" not in self._init_locks:
                self._init_locks["context_augmenter"] = asyncio.Lock()

            async with self._init_locks["context_augmenter"]:
                if self._context_augmenter is None:
                    from src.application.context import ContextAugmenter
                    from src.application.context.fraction_classifier import FractionalQueryClassifier
                    from src.application.models.embeddings import E5EmbeddingModel

                    # Initialize E5 embedding model
                    embedding_model = E5EmbeddingModel(
                        model_name="intfloat/multilingual-e5-large-instruct",
                        cache_dir="models/embeddings/e5_cache"
                    )
                    embedding_model.load()

                    # Initialize fractional query classifier
                    fractional_classifier = FractionalQueryClassifier(
                        model_path="models/context/fractional_classifier.pkl",
                        embedding_model=embedding_model
                    )

                    # Initialize context augmenter
                    self._context_augmenter = ContextAugmenter(
                        classifier=fractional_classifier,
                        mappings_file="config/context_mappings.json"
                    )

        return self._context_augmenter
    
    async def get_summarizer(self) -> Any:
        """Get tenant-specific summarizer (lazy init)"""
        if self._summarizer is None:
            if "summarizer" not in self._init_locks:
                self._init_locks["summarizer"] = asyncio.Lock()
            
            async with self._init_locks["summarizer"]:
                if self._summarizer is None:
                    from src.application.summarization import PassthroughSummarizer
                    self._summarizer = PassthroughSummarizer()
        
        return self._summarizer
    
    async def get_nlp_service(self) -> Any:
        """Get tenant-specific NLP service (lazy init)"""
        if self._nlp_service is None:
            logger.info(f"[CONTEXT] Initializing NLP service for tenant: {self.config.tenant_id}")
            if "nlp_service" not in self._init_locks:
                self._init_locks["nlp_service"] = asyncio.Lock()

            async with self._init_locks["nlp_service"]:
                if self._nlp_service is None:
                    from src.application.nlp_service import AsyncNLPPipeline, NLPServiceWithCircuitBreaker
                    from src.application.models import ClassifierModel, SemanticSearchModel, E5EmbeddingModel, ModelConfig

                    try:
                        # Create E5 embedding model
                        logger.info("[NLP] Loading E5 embedding model...")
                        embedding_config = ModelConfig(
                            name="e5_embedding",
                            type="embedding",
                            implementation="E5EmbeddingModel",
                            config={
                                "name": "intfloat/multilingual-e5-large-instruct",
                                "cache_dir": "models/embeddings/e5_cache",
                                "device": "cpu"
                            }
                        )
                        embedding_model = E5EmbeddingModel(embedding_config)
                        embedding_model.load()
                        logger.info("✅ E5 embedding model loaded successfully")

                        # Create classifier
                        logger.info("[NLP] Loading classifier model...")
                        classifier_config = ModelConfig(
                            name="classifier",
                            type="classifier",
                            implementation="ClassifierModel",
                            config={
                                "model_path": "models/classification/model.pkl",
                                "label_encoder_path": "models/classification/label_encoder.pkl"
                            }
                        )
                        classifier = ClassifierModel(classifier_config, embedding_model)
                        classifier.load()
                        logger.info("✅ Classifier loaded successfully")

                        # Create semantic searcher
                        logger.info("[NLP] Loading FAISS indices...")
                        search_config = ModelConfig(
                            name="searcher",
                            type="search",
                            implementation="SemanticSearchModel",
                            config={
                                "indices_dir": "models/semantic_search/faiss_indices",
                                "metadata_path": "models/semantic_search/cluster_metadata.json"
                            }
                        )
                        searcher = SemanticSearchModel(search_config, embedding_model)
                        searcher.load()
                        logger.info("✅ FAISS indices loaded successfully")

                        # Create NLP pipeline with circuit breaker
                        pipeline = AsyncNLPPipeline(classifier, searcher, self.config.confidence_threshold)
                        self._nlp_service = NLPServiceWithCircuitBreaker(pipeline)
                        logger.info(f"✅ NLP service initialized (confidence threshold: {self.config.confidence_threshold})")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize NLP service: {e}", exc_info=True)
                        raise

        return self._nlp_service
    
    async def get_policy_engine(self) -> Any:
        """Get tenant-specific policy engine (lazy init)"""
        if self._policy_engine is None:
            logger.info(f"[CONTEXT] Initializing policy engine for tenant: {self.config.tenant_id}")
            if "policy_engine" not in self._init_locks:
                self._init_locks["policy_engine"] = asyncio.Lock()
            
            async with self._init_locks["policy_engine"]:
                if self._policy_engine is None:
                    try:
                        logger.info("[POLICY] Initializing policy engine...")
                        from src.application.policy import PolicyEngine
                        from src.application.policy.handlers import (
                            ActiveFormHandler, FormTriggerHandler,
                            HighConfidenceFAQHandler, ClarificationFallbackHandler,
                            EscalationHandler
                        )
                        from src.application.policy.faq_policy import FAQPolicy
                        from src.application.policy.clarification_policy import ClarificationPolicy
                        from src.application.policy.form_policy import FormPolicy
                        from src.application.forms.form_runner import FormRunner

                        # Create policy instances
                        faq_policy = FAQPolicy(confidence_threshold=self.config.confidence_threshold)
                        clarification_policy = ClarificationPolicy(
                            min_confidence=self.config.clarification_threshold
                        )

                        # Create form-related components
                        form_registry = await self.get_form_registry()
                        form_runner = FormRunner()
                        form_policy = FormPolicy(form_runner, form_registry)

                        # Create handlers
                        handler1 = ActiveFormHandler(form_policy)
                        handler2 = FormTriggerHandler(form_registry)
                        handler3 = HighConfidenceFAQHandler(faq_policy)
                        handler4 = ClarificationFallbackHandler(clarification_policy)
                        handler5 = EscalationHandler()

                        # Link the chain
                        handler1.set_next(handler2)
                        handler2.set_next(handler3)
                        handler3.set_next(handler4)
                        handler4.set_next(handler5)

                        self._policy_engine = PolicyEngine(handler_chain=handler1)
                        logger.info("✅ Policy engine initialized")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize policy engine: {e}", exc_info=True)
                        raise
        
        return self._policy_engine
    
    async def get_form_registry(self) -> Any:
        """Get tenant-specific form registry (lazy init)"""
        if self._form_registry is None:
            if "form_registry" not in self._init_locks:
                self._init_locks["form_registry"] = asyncio.Lock()
            
            async with self._init_locks["form_registry"]:
                if self._form_registry is None:
                    from src.application.forms import FormRegistry
                    from src.application.forms.examples import (
                        NIDStatusCheckForm, CorrectionRequestForm,
                        HotelBookingForm, BankAccountVerificationForm
                    )
                    
                    registry = FormRegistry()
                    # Register all forms
                    registry.register(NIDStatusCheckForm())
                    registry.register(CorrectionRequestForm())
                    registry.register(HotelBookingForm())
                    registry.register(BankAccountVerificationForm())
                    
                    self._form_registry = registry
        
        return self._form_registry
    
    async def cleanup(self) -> None:
        """Cleanup resources when context is done"""
        # Close Redis connections, release locks, etc.
        if self._redis:
            await self._redis.close()
        
        # Unload models
        if self._model_registry:
            await self._model_registry.unload_all()


class TenantContextFactory:
    """Factory for creating tenant contexts"""
    
    def __init__(self):
        self._tenant_configs: Dict[str, TenantConfig] = {}
        self._session_stores: Dict[str, Any] = {}  # tenant_id → SessionStore
    
    def register_tenant(self, config: TenantConfig) -> None:
        """Register a tenant configuration"""
        self._tenant_configs[config.tenant_id] = config
    
    def register_session_store(self, tenant_id: str, store: Any) -> None:
        """Register shared session store for tenant"""
        self._session_stores[tenant_id] = store
        logger.info(f"Registered session store for tenant: {tenant_id}")
    
    def get_session_store(self, tenant_id: str) -> Any:
        """Get shared session store for tenant"""
        return self._session_stores.get(tenant_id)
    
    async def create_context(
        self,
        tenant_id: str,
        correlation_id: Optional[str] = None
    ) -> TenantContext:
        """Create a new tenant context for a request"""
        config = self._tenant_configs.get(tenant_id)
        
        if config is None:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        return TenantContext(
            tenant_id=tenant_id,
            config=config,
            factory=self,  # Pass factory reference for shared resources
            correlation_id=correlation_id,
        )
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self._tenant_configs.get(tenant_id)
