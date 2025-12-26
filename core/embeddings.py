"""
Embedding service using Azure OpenAI
Uses the same endpoint as your GPT models
"""

from typing import List
from openai import AzureOpenAI
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Azure OpenAI embedding service"""
    
    def __init__(self):
        self.use_api = config.embedding.use_api
        self.dimension = config.embedding.dimension
        self.deployment = config.embedding.azure_deployment
        
        if self.use_api:
            # Use same Azure OpenAI client as your LLM
            self.client = AzureOpenAI(
                api_key=config.azure.api_key,
                api_version=config.azure.api_version,
                azure_endpoint=config.azure.endpoint
            )
            logger.info(f"Embedding service: Azure OpenAI ({self.deployment}, dim={self.dimension})")
        else:
            self.client = None
            self.local_model = None
            logger.info(f"Embedding service: Local fallback")
    
    def _load_local_model(self):
        """Load local model as fallback"""
        if self.local_model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model...")
            # Use a model with same dimension
            self.local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            logger.info("Local model loaded")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _call_azure_api(self, texts: List[str]) -> List[List[float]]:
        """Call Azure OpenAI embeddings API"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.deployment
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return embeddings
        
        except Exception as e:
            logger.error(f"Azure embedding API error: {e}")
            raise
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally"""
        if self.local_model is None:
            self._load_local_model()
        
        embeddings = self.local_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed batch of texts
        
        Azure supports up to 2048 texts per request, but we batch to be safe
        """
        if not texts:
            return []
        
        # Truncate texts (Azure handles this but we do it for consistency)
        max_chars = config.embedding.max_length * 4
        truncated = [text[:max_chars] for text in texts]
        
        try:
            if self.use_api:
                # Process in batches
                all_embeddings = []
                for i in range(0, len(truncated), batch_size):
                    batch = truncated[i:i + batch_size]
                    embeddings = self._call_azure_api(batch)
                    all_embeddings.extend(embeddings)
                    
                    if i + batch_size < len(truncated):
                        logger.debug(f"Processed {i + batch_size}/{len(truncated)} texts")
                
                return all_embeddings
            else:
                return self._embed_local(truncated)
        
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            
            # Fallback to local
            if self.use_api:
                logger.warning("Azure API failed, falling back to local model...")
                return self._embed_local(truncated)
            
            raise
    
    def test_connection(self) -> bool:
        """Test embedding service"""
        try:
            test_text = "This is a test."
            logger.info("Testing Azure embedding service...")
            
            embedding = self.embed_text(test_text)
            
            if len(embedding) == self.dimension:
                logger.info("✅ Embedding test successful")
                logger.info(f"   Deployment: {self.deployment}")
                logger.info(f"   Dimension: {len(embedding)}")
                return True
            else:
                logger.error(f"Dimension mismatch: expected {self.dimension}, got {len(embedding)}")
                return False
        
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            return False