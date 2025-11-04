"""
Standalone Semantic Search Component implementation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import time

from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

# Reuse existing models
from src.application.models.search import SemanticSearchModel
from src.application.models.embeddings import E5EmbeddingModel
from src.application.models.base import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchConfig(ComponentConfig):
    """Configuration for Semantic Search component"""
    indices_dir: str = "models/semantic_search/faiss_indices"
    metadata_path: str = "models/semantic_search/cluster_metadata.json"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_cache_dir: str = "./models/embeddings/e5_cache"
    top_k: int = 5


@dataclass
class SemanticSearchInput(ComponentInput):
    """Input for Semantic Search component"""
    cluster: Optional[str] = None  # Optional cluster to search in (None = merged search)
    top_k: Optional[int] = None  # Override default top_k


@dataclass
class SearchResultItem:
    """Single search result"""
    question: str
    cluster: str
    tag: Optional[str]
    score: float
    search_type: str = "merged"  # "cluster" or "merged"


@dataclass
class SemanticSearchOutput(ComponentOutput):
    """Output from Semantic Search component"""
    results: List[SearchResultItem] = field(default_factory=list)
    search_type: str = "merged"  # "cluster" or "merged"
    results_count: int = 0

    def __post_init__(self):
        # Ensure data contains results
        if self.data is None or not isinstance(self.data, dict):
            self.data = {
                "results": [
                    {
                        "question": r.question,
                        "cluster": r.cluster,
                        "tag": r.tag,
                        "score": r.score,
                        "search_type": r.search_type,
                    }
                    for r in self.results
                ],
                "search_type": self.search_type,
                "results_count": self.results_count,
            }


class SemanticSearchComponent(Component[SemanticSearchInput, SemanticSearchOutput, SemanticSearchConfig]):
    """
    Standalone Semantic Search Component.

    Features:
    - FAISS-based semantic search using E5 embeddings
    - Cluster-specific or merged search
    - Top-K results with similarity scores
    - Can run independently or in pipeline

    Example usage:
        config = SemanticSearchConfig(device="cpu")
        search = SemanticSearchComponent(config)
        await search.initialize()

        input_data = SemanticSearchInput(
            text="আমার NID স্ট্যাটাস কি?",
            cluster="nid_status"  # Optional
        )
        output = await search.process(input_data)
        print(output.results)
    """

    def __init__(self, config: SemanticSearchConfig):
        super().__init__(config)
        self.embedding_model: Optional[E5EmbeddingModel] = None
        self.search_model: Optional[SemanticSearchModel] = None

    async def initialize(self) -> None:
        """Initialize embedding model and FAISS indices"""
        if self._initialized:
            logger.info("Semantic Search component already initialized")
            return

        logger.info("Initializing Semantic Search component...")
        start_time = time.time()

        # Initialize embedding model
        embedding_config = ModelConfig(config={
            "name": self.config.embedding_model,
            "cache_dir": self.config.embedding_cache_dir,
            "device": self.config.device,
        })
        self.embedding_model = E5EmbeddingModel(embedding_config)
        self.embedding_model.load()

        # Initialize search model
        search_config = ModelConfig(config={
            "indices_dir": self.config.indices_dir,
            "metadata_path": self.config.metadata_path,
        })
        self.search_model = SemanticSearchModel(search_config, self.embedding_model)
        self.search_model.load()

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Semantic Search component initialized in {elapsed:.2f}ms")

    async def process(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
        """Search for similar questions"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Determine top_k
            top_k = input_data.top_k or self.config.top_k

            # Run search (CPU-bound, but in async context)
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=2)
            loop = asyncio.get_event_loop()

            results_raw = await loop.run_in_executor(
                executor,
                self.search_model.search,
                input_data.text,
                input_data.cluster,
                top_k
            )

            # Convert to typed results
            results = [
                SearchResultItem(
                    question=res["question"],
                    cluster=res["cluster"],
                    tag=res.get("tag"),
                    score=res["score"],
                    search_type=res.get("search_type", "merged"),
                )
                for res in results_raw
            ]

            search_type = results[0].search_type if results else "merged"
            results_count = len(results)

            processing_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchOutput(
                success=True,
                data={
                    "results": [
                        {
                            "question": r.question,
                            "cluster": r.cluster,
                            "tag": r.tag,
                            "score": r.score,
                            "search_type": r.search_type,
                        }
                        for r in results
                    ],
                    "search_type": search_type,
                    "results_count": results_count,
                },
                results=results,
                search_type=search_type,
                results_count=results_count,
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

        except Exception as e:
            logger.error(f"Semantic Search processing failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchOutput(
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

    def get_name(self) -> str:
        """Return component name"""
        return "semantic_search"

    async def health_check(self) -> Dict[str, Any]:
        """Health check for Semantic Search component"""
        base_health = await super().health_check()

        search_specific = {
            "embedding_model_loaded": self.embedding_model.is_loaded() if self.embedding_model else False,
            "search_model_loaded": self.search_model.is_loaded() if self.search_model else False,
            "indices_count": len(self.search_model.indices) if self.search_model else 0,
        }

        return {**base_health, **search_specific}


# CLI entry point
async def main():
    """CLI entry point for standalone Semantic Search component"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Standalone Semantic Search Component")
    parser.add_argument("--text", type=str, help="Text to search")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    parser.add_argument("--indices-dir", type=str, default="models/semantic_search/faiss_indices")
    parser.add_argument("--metadata-path", type=str, default="models/semantic_search/cluster_metadata.json")
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cluster", type=str, help="Cluster to search in (optional)")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    # Create config
    config = SemanticSearchConfig(
        indices_dir=args.indices_dir,
        metadata_path=args.metadata_path,
        embedding_model=args.embedding_model,
        device=args.device,
        top_k=args.top_k,
    )

    # Initialize component
    component = SemanticSearchComponent(config)
    await component.initialize()

    # Prepare input
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_data = SemanticSearchInput(**input_json)
    elif args.text:
        input_data = SemanticSearchInput(text=args.text, cluster=args.cluster)
    else:
        print("Error: Must provide --text or --input-file", file=sys.stderr)
        sys.exit(1)

    # Process
    output = await component.process(input_data)

    # Write output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output.to_json())
    else:
        print(output.to_json())

    # Exit with appropriate code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
