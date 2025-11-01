"""
NLP Service - Async-first NLP pipeline with no blocking operations.

Replaces the problematic synchronous-in-async pattern from original codebase.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.tracing import trace_async, Tracer
from src.infrastructure.circuit_breaker import with_circuit_breaker


# Thread pool for CPU-bound operations (classification, embeddings)
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class ClassificationResult:
    """Type-safe classification result"""
    cluster: str
    confidence: float


@dataclass
class SearchResult:
    """Type-safe search result"""
    question: str
    cluster: str
    tag: Optional[str]
    score: float


@dataclass
class NLPResult:
    """Complete NLP pipeline result"""
    success: bool
    classification: List[ClassificationResult]
    search_results: List[SearchResult]
    strategy: str  # "single_cluster" or "merged_fallback"
    confidence: float
    processing_time_ms: float
    error: Optional[str] = None


class AsyncNLPPipeline:
    """
    Async-first NLP pipeline with proper concurrency.
    
    NO BLOCKING OPERATIONS in async functions!
    All CPU-bound work (embeddings, classification) runs in thread pool.
    
    Integrates with actual Classifier and Searcher models from registry.
    """
    
    def __init__(
        self,
        classifier,  # ClassifierModel from models layer
        searcher,    # SemanticSearchModel from models layer
        confidence_threshold: float = 0.6,
    ):
        self.classifier = classifier
        self.searcher = searcher
        self.confidence_threshold = confidence_threshold
    
    @trace_async("nlp.pipeline.run")
    async def run(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> NLPResult:
        """
        Run full NLP pipeline asynchronously.
        
        Steps:
        1. Classify query (in thread pool)
        2. Search based on confidence:
           - High confidence → cluster index
           - Low confidence → merged index
        3. Return results with metadata
        """
        span = Tracer.current_span()
        if span:
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            if tenant_id:
                span.set_attribute("tenant_id", tenant_id)
        
        try:
            # Step 1: Classify (CPU-bound, run in thread pool)
            classification = await self._classify_async(query, top_k=3)
            
            if span:
                span.set_attribute("top_cluster", classification[0].cluster)
                span.set_attribute("confidence", classification[0].confidence)
            
            # Step 2: Determine search strategy
            top_confidence = classification[0].confidence
            
            if top_confidence >= self.confidence_threshold:
                # High confidence: search in predicted cluster
                strategy = "single_cluster"
                cluster = classification[0].cluster
            else:
                # Low confidence: search in merged index
                strategy = "merged_fallback"
                cluster = None
            
            if span:
                span.set_attribute("search_strategy", strategy)
                span.set_attribute("search_cluster", cluster or "merged")
            
            # Step 3: Search (CPU-bound, run in thread pool)
            search_results = await self._search_async(
                query,
                cluster=cluster,
                top_k=top_k
            )
            
            if span:
                span.set_attribute("results_count", len(search_results))
                span.add_event("nlp_pipeline_completed", {
                    "strategy": strategy,
                    "results": len(search_results)
                })
            
            return NLPResult(
                success=True,
                classification=classification,
                search_results=search_results,
                strategy=strategy,
                confidence=top_confidence,
                processing_time_ms=0,  # Will be set by tracer
            )
        
        except Exception as e:
            # Log detailed error information
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"NLP pipeline failed: {e}", exc_info=True)
            logger.error(f"Query that caused failure: '{query}'")
            
            if span:
                span.set_attribute("error", str(e))
                span.set_attribute("error_type", type(e).__name__)
            
            return NLPResult(
                success=False,
                classification=[],
                search_results=[],
                strategy="error",
                confidence=0.0,
                processing_time_ms=0,
                error=str(e),
            )
    
    @trace_async("nlp.classify")
    async def _classify_async(
        self,
        query: str,
        top_k: int = 3
    ) -> List[ClassificationResult]:
        """
        Classify query asynchronously.
        
        Runs CPU-bound classification in thread pool to avoid blocking event loop.
        Uses actual ClassifierModel with E5 embeddings.
        """
        loop = asyncio.get_event_loop()
        
        # Ensure classifier is loaded
        if not self.classifier.is_loaded():
            await loop.run_in_executor(_executor, self.classifier.load)
        
        # Run in thread pool (non-blocking)
        predictions = await loop.run_in_executor(
            _executor,
            self.classifier.predict,
            query,
            top_k
        )
        
        # Convert to type-safe results
        return [
            ClassificationResult(
                cluster=pred["cluster"],
                confidence=pred["confidence"]
            )
            for pred in predictions
        ]
    
    @trace_async("nlp.search")
    async def _search_async(
        self,
        query: str,
        cluster: Optional[str],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search asynchronously.
        
        Runs FAISS search in thread pool to avoid blocking event loop.
        Uses actual SemanticSearchModel with E5 embeddings.
        """
        loop = asyncio.get_event_loop()
        
        # Ensure searcher is loaded
        if not self.searcher.is_loaded():
            await loop.run_in_executor(_executor, self.searcher.load)
        
        # Run in thread pool (non-blocking)
        results = await loop.run_in_executor(
            _executor,
            self.searcher.search,
            query,
            cluster,
            top_k
        )
        
        # Convert to type-safe results
        return [
            SearchResult(
                question=result["question"],
                cluster=result["cluster"],
                tag=result.get("tag"),
                score=result["score"]
            )
            for result in results
        ]
    
    @trace_async("nlp.embed")
    async def embed_async(self, texts: List[str]) -> Any:
        """
        Generate embeddings asynchronously.
        
        Used by classifier and searcher internally.
        """
        loop = asyncio.get_event_loop()
        
        # Run embedding generation in thread pool
        embeddings = await loop.run_in_executor(
            _executor,
            self._embed_sync,
            texts
        )
        
        return embeddings
    
    def _embed_sync(self, texts: List[str]) -> Any:
        """Synchronous embedding generation (runs in thread pool)"""
        # This would call the actual embedding model
        # For now, placeholder
        return None


class NLPServiceWithCircuitBreaker:
    """
    NLP service with circuit breaker for external dependencies.
    
    If embedding service goes down, circuit breaker prevents cascading failures.
    """
    
    def __init__(self, pipeline: AsyncNLPPipeline):
        self.pipeline = pipeline
    
    @with_circuit_breaker(
        name="nlp_pipeline",
        failure_threshold=5,
        recovery_timeout=30
    )
    @trace_async("nlp_service.process")
    async def process(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5
    ) -> NLPResult:
        """
        Process query with circuit breaker protection.
        
        If pipeline fails repeatedly, circuit opens and fails fast.
        """
        return await self.pipeline.run(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id
        )
