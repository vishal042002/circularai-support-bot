"""
Pinecone vector store operations
"""

from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Pinecone vector store wrapper"""
    
    def __init__(self, embedding_dimension: int = None):
        self.pinecone_config = config.pinecone
        self.embedding_dim = embedding_dimension or config.embedding.dimension
        
        self.client = Pinecone(api_key=self.pinecone_config.api_key)
        self.index = self._initialize_index()
        
        logger.info(f"Vector store: {self.pinecone_config.index_name}")
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        index_name = self.pinecone_config.index_name
        existing_indexes = self.client.list_indexes().names()
        
        if index_name in existing_indexes:
            # Validate dimensions
            index_info = self.client.describe_index(index_name)
            
            if index_info.dimension != self.embedding_dim:
                raise ValueError(
                    f"Dimension mismatch: index={index_info.dimension}, "
                    f"model={self.embedding_dim}. Please recreate index."
                )
            
            logger.info(f"Using existing index (dim: {index_info.dimension})")
        else:
            # Create new index
            logger.info(f"Creating new index (dim: {self.embedding_dim})...")
            self.client.create_index(
                name=index_name,
                dimension=self.embedding_dim,
                metric=self.pinecone_config.metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_config.cloud,
                    region=self.pinecone_config.region
                )
            )
            logger.info("Index created successfully")
        
        return self.client.Index(index_name)
    
    def upsert(self, vectors: List[Dict], batch_size: int = 100):
        """Upsert vectors to index"""
        if not vectors:
            return
        
        # Batch upsert
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return [
            {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "score": float(match.score),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
        ]
    
    def delete_all(self):
        """Delete all vectors"""
        self.index.delete(delete_all=True)
        logger.warning("All vectors deleted")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return self.index.describe_index_stats()