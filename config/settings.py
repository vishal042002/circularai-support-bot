"""
Centralized configuration management
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# For Streamlit Cloud deployment
try:
    import streamlit as st
    STREAMLIT_SECRETS = st.secrets
except:
    STREAMLIT_SECRETS = None


def get_env(key: str, default: str = None) -> str:
    """Get environment variable from .env or Streamlit secrets"""
    # Try Streamlit secrets first (when deployed)
    if STREAMLIT_SECRETS is not None:
        try:
            if key in STREAMLIT_SECRETS:
                return str(STREAMLIT_SECRETS[key])
        except:
            pass  # Streamlit secrets not available, fall through to .env
    
    # Fall back to environment variables (local dev)
    return os.getenv(key, default)


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration"""
    endpoint: str
    api_key: str
    deployment_name: str
    vision_deployment_name: str
    api_version: str = "2024-08-01-preview"


class PineconeConfig(BaseModel):
    """Pinecone configuration"""
    api_key: str
    index_name: str
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"


class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    use_api: bool
    dimension: int
    max_length: int = 512
    azure_deployment: str


class ChunkingConfig(BaseModel):
    """Document chunking configuration"""
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    respect_sections: bool = True
    preserve_steps: bool = True


class SearchConfig(BaseModel):
    """Search configuration"""
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k: int = 5


class SystemConfig(BaseModel):
    """Complete system configuration"""
    azure: AzureOpenAIConfig
    pinecone: PineconeConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    search: SearchConfig
    
    checkpoint_db: str = "data/checkpoints/conversations.db"
    log_level: str
    
    @classmethod
    def load(cls) -> "SystemConfig":
        """Load configuration from environment or Streamlit secrets"""
        return cls(
            azure=AzureOpenAIConfig(
                endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
                api_key=get_env("AZURE_OPENAI_API_KEY"),
                deployment_name=get_env("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                vision_deployment_name=get_env("AZURE_VISION_DEPLOYMENT_NAME", "gpt-4o")
            ),
            pinecone=PineconeConfig(
                api_key=get_env("PINECONE_API_KEY"),
                index_name=get_env("PINECONE_INDEX_NAME", "circularai-support")
            ),
            embedding=EmbeddingConfig(
                model_name=get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
                use_api=get_env("USE_EMBEDDING_API", "true").lower() == "true",
                dimension=int(get_env("AZURE_EMBEDDING_DIMENSION", "1536")),
                azure_deployment=get_env("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
            ),
            chunking=ChunkingConfig(),
            search=SearchConfig(),
            log_level=get_env("LOG_LEVEL", "INFO")
        )
    
    def setup_langsmith(self):
        """Setup LangSmith tracing"""
        langsmith_key = get_env("LANGSMITH_API_KEY")
        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_PROJECT"] = get_env("LANGSMITH_PROJECT", "circularai-rag")


# Global config instance
config = SystemConfig.load()
config.setup_langsmith()