"""
Hybrid search combining semantic and keyword search
"""

from typing import List, Dict
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridSearch:
    """Hybrid search combining semantic (embeddings) and keyword (BM25)"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        self.embedder = embedding_service
        self.vector_store = vector_store
        
        self.semantic_weight = config.search.semantic_weight
        self.bm25_weight = config.search.bm25_weight
        self.top_k = config.search.top_k
        
        # BM25 index
        self.bm25_index = None
        self.bm25_corpus = []
        
        # Auto-build BM25 index from Pinecone on startup
        self._initialize_bm25_from_pinecone()
        
        logger.info(f"Hybrid search: semantic={self.semantic_weight}, bm25={self.bm25_weight}")
    
    def _initialize_bm25_from_pinecone(self):
        """Initialize BM25 index by fetching all vectors from Pinecone"""
        try:
            stats = self.vector_store.get_stats()
            total_vectors = stats.get('total_vector_count', 0)

            if total_vectors == 0:
                logger.warning("No vectors in Pinecone, BM25 index will be empty")
                return

            # Try to build from JSON first (faster)
            from pathlib import Path
            import json

            processed_dir = Path("data/processed")
            chunks_file = list(processed_dir.glob("*_chunks.json"))

            if chunks_file:
                with open(chunks_file[0], 'r') as f:
                    chunks = json.load(f)

                # Verify JSON count matches Pinecone count
                if len(chunks) == total_vectors:
                    logger.info(f"Building BM25 index from {len(chunks)} JSON chunks (verified sync)...")
                    documents = [
                        {
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "metadata": chunk["metadata"]
                        }
                        for chunk in chunks
                    ]
                    self.build_bm25_index(documents)
                    return
                else:
                    logger.warning(
                        f"JSON/Pinecone mismatch: JSON={len(chunks)}, Pinecone={total_vectors}. "
                        f"Fetching from Pinecone instead..."
                    )

            # Fallback: Fetch from Pinecone
            logger.info(f"Building BM25 index from {total_vectors} Pinecone vectors...")
            documents = self._fetch_all_from_pinecone()

            if documents:
                self.build_bm25_index(documents)
            else:
                logger.warning("No valid documents available for BM25")

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")

    def _fetch_all_from_pinecone(self) -> List[Dict]:
        """Fetch all documents from Pinecone for BM25 indexing"""
        stats = self.vector_store.get_stats()
        total_vectors = stats.get('total_vector_count', 0)

        if total_vectors == 0:
            return []

        # Query with dummy vector to get all results
        # This is a workaround since Pinecone doesn't have direct fetch_all()
        dummy_vector = [0.0] * self.embedder.dimension
        batch_size = 10000  # Pinecone max top_k

        results = self.vector_store.index.query(
            vector=dummy_vector,
            top_k=min(total_vectors, batch_size),
            include_metadata=True
        )

        return [
            {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
            if match.metadata.get("text", "").strip()  # Only valid text
        ]

    def build_bm25_index(self, documents: List[Dict]):
        """Build BM25 index from documents"""
        self.bm25_corpus = documents
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict]:
        """Semantic search using embeddings"""
        top_k = top_k or self.top_k
        
        query_vector = self.embedder.embed_text(query)
        results = self.vector_store.search(query_vector, top_k=top_k * 2)
        
        return results
    
    def bm25_search(self, query: str, top_k: int = None) -> List[Dict]:
        """Keyword search using BM25"""
        if self.bm25_index is None:
            logger.warning("BM25 index not built")
            return []
        
        top_k = top_k or self.top_k
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.bm25_corpus[idx]
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(scores[idx]),
                    "metadata": doc["metadata"]
                })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_results: List[Dict] = None,
        bm25_results: List[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid search with weighted score fusion
        
        Formula: final_score = semantic_weight * semantic_score + bm25_weight * bm25_score
        """
        top_k = top_k or self.top_k
        
        # Get results if not provided
        if semantic_results is None:
            semantic_results = self.semantic_search(query, top_k)
        if bm25_results is None:
            bm25_results = self.bm25_search(query, top_k)
        
        # Normalize scores to [0, 1] while preserving relative quality
        def normalize(results, search_type="semantic"):
            if not results:
                return []
            scores = [r["score"] for r in results]

            # Use different normalization strategies based on search type
            if search_type == "semantic":
                # Semantic scores are cosine similarity (already 0-1 range)
                # Just use them directly - no normalization needed!
                for r in results:
                    r["normalized_score"] = r["score"]
            else:
                # BM25 scores: use min-max but preserve absolute quality
                max_score, min_score = max(scores), min(scores)
                range_score = max_score - min_score if max_score != min_score else 1

                for r in results:
                    # Min-max normalization
                    normalized = (r["score"] - min_score) / range_score
                    # Scale down if max score is low (indicates poor overall quality)
                    quality_factor = min(max_score / 10.0, 1.0)  # Assume 10+ is good BM25 score
                    r["normalized_score"] = normalized * quality_factor

            return results
        
        semantic_results = normalize(semantic_results, search_type="semantic")
        bm25_results = normalize(bm25_results, search_type="bm25")
        
        # Combine scores
        combined = defaultdict(lambda: {"semantic": 0, "bm25": 0, "doc": None})
        
        for r in semantic_results:
            combined[r["id"]]["semantic"] = r["normalized_score"]
            combined[r["id"]]["doc"] = r
        
        for r in bm25_results:
            combined[r["id"]]["bm25"] = r["normalized_score"]
            if combined[r["id"]]["doc"] is None:
                combined[r["id"]]["doc"] = r
        
        # Calculate final scores
        final_results = []
        for doc_id, scores in combined.items():
            final_score = (
                self.semantic_weight * scores["semantic"] +
                self.bm25_weight * scores["bm25"]
            )

            result = scores["doc"]

            # BOOST: Penalize screenshot-heavy chunks
            text = result.get("text", "")
            if text.startswith("[Related Screenshots:]"):
                # Reduce score for screenshot descriptions by 30%
                final_score *= 0.7
                logger.debug(f"Applied screenshot penalty to chunk {doc_id[:50]}")

            result["hybrid_score"] = final_score
            result["semantic_score"] = scores["semantic"]
            result["bm25_score"] = scores["bm25"]
            final_results.append(result)
        
        # Sort by hybrid score
        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Debug logging for top result
        if final_results:
            top = final_results[0]
            logger.debug(
                f"Top result - Hybrid: {top['hybrid_score']:.3f} "
                f"(Semantic: {top['semantic_score']:.3f}, BM25: {top['bm25_score']:.3f})"
            )

        return final_results[:top_k]