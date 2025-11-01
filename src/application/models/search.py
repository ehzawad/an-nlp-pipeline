"""Semantic search model implementations."""

import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseModel, ModelConfig
from .embeddings import EmbeddingModel
import logging

logger = logging.getLogger(__name__)


class SemanticSearchModel(BaseModel):
    """FAISS-based semantic search."""

    def __init__(self, config: ModelConfig, embedding_model: EmbeddingModel):
        """
        Initialize search model with embedding dependency.
        
        Args:
            config: Model configuration
            embedding_model: Embedding model for query encoding
        """
        super().__init__(config)
        self.embedding_model = embedding_model
        self.indices = {}  # cluster -> faiss index
        self.metadata = {}  # cluster -> metadata
        self.merged_index = None
        self.merged_metadata = None
        
        self.indices_dir = config.config.get(
            "indices_dir",
            "models/semantic_search/faiss_indices"
        )
        self.metadata_path = config.config.get(
            "metadata_path",
            "models/semantic_search/cluster_metadata.json"
        )

    def load(self) -> None:
        """Load FAISS indices and metadata."""
        if self._loaded:
            logger.info("Search model already loaded")
            return

        logger.info(f"Loading FAISS indices from {self.indices_dir}")

        try:
            indices_path = Path(self.indices_dir)
            
            # Load per-cluster indices
            for index_file in indices_path.glob("*.index"):
                cluster_name = index_file.stem
                
                # Skip merged index for now
                if cluster_name == "merged_all":
                    continue
                
                index = faiss.read_index(str(index_file))
                self.indices[cluster_name] = index
                logger.debug(f"Loaded index for cluster: {cluster_name}")
            
            # Load merged index
            merged_path = indices_path / "merged_all.index"
            if merged_path.exists():
                self.merged_index = faiss.read_index(str(merged_path))
                logger.info("Loaded merged_all index")

            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_json = json.load(f)

                # Handle the actual format: {"cluster_to_questions": {"cluster": ["q1", "q2", ...]}}
                if "cluster_to_questions" in metadata_json:
                    cluster_to_questions = metadata_json["cluster_to_questions"]
                    question_to_tag = metadata_json.get("question_to_tag", {})

                    # Convert to the expected format with tags
                    all_metadata = []
                    for cluster, questions in cluster_to_questions.items():
                        # Store questions by cluster with their tags
                        self.metadata[cluster] = [
                            {
                                "question": q, 
                                "cluster": cluster,
                                "tag": question_to_tag.get(q)  # Add tag from mapping
                            } 
                            for q in questions
                        ]
                        all_metadata.extend(self.metadata[cluster])

                    # Store all for merged search
                    self.merged_metadata = all_metadata
                    logger.debug(f"Loaded {len(question_to_tag)} question-to-tag mappings")
                else:
                    # Fallback: assume it's already in the correct format
                    all_metadata = metadata_json
                    for item in all_metadata:
                        cluster = item.get('cluster', 'unknown')
                        if cluster not in self.metadata:
                            self.metadata[cluster] = []
                        self.metadata[cluster].append(item)
                    self.merged_metadata = all_metadata

            self._loaded = True
            logger.info(
                f"Search model loaded: {len(self.indices)} cluster indices, "
                f"{len(self.merged_metadata)} total questions"
            )

        except Exception as e:
            logger.error(f"Failed to load search model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if search model is loaded."""
        return self._loaded

    def search(
        self,
        query: str,
        cluster: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar questions.
        
        Args:
            query: User query
            cluster: Optional cluster to search in (None = merged search)
            top_k: Number of results to return
            
        Returns:
            List of search results with question, cluster, tag, score
        """
        if not self.is_loaded():
            self.load()

        # Get query embedding
        query_embedding = self.embedding_model.embed(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Choose index and metadata
        if cluster and cluster in self.indices:
            # Cluster-specific search
            index = self.indices[cluster]
            metadata = self.metadata.get(cluster, [])
            search_type = "cluster"
        else:
            # Merged search
            index = self.merged_index
            metadata = self.merged_metadata
            search_type = "merged"

        if index is None or not metadata:
            logger.warning(f"No index or metadata for search type: {search_type}")
            return []

        # Search
        scores, indices = index.search(query_embedding, min(top_k, len(metadata)))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            
            item = metadata[idx]
            results.append({
                "question": item.get("question", ""),
                "cluster": item.get("cluster", ""),
                "tag": item.get("tag"),
                "score": float(score),
                "search_type": search_type
            })

        return results

