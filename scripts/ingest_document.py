"""
Ingest DOCX document through complete pipeline
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path

from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from core.search import HybridSearch
from ingestion.pipeline import IngestionPipeline
from utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest DOCX document")
    parser.add_argument(
        "docx_path",
        type=str,
        help="Path to DOCX file"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image processing (faster but less accurate)"
    )
    
    args = parser.parse_args()
    
    # Validate file
    docx_path = Path(args.docx_path)
    if not docx_path.exists():
        print(f"❌ File not found: {docx_path}")
        return 1
    
    if docx_path.suffix.lower() != '.docx':
        print(f"❌ File must be .docx format")
        return 1
    
    # Initialize components
    print("\nInitializing RAG components...")
    embedder = EmbeddingService()
    vector_store = VectorStore()
    llm = LLMService()
    search = HybridSearch(embedder, vector_store)
    
    # Create pipeline
    pipeline = IngestionPipeline(embedder, vector_store, search)
    
    # Ingest document
    try:
        stats = pipeline.ingest_docx(
            str(docx_path),
            process_images=not args.no_images
        )
        
        print("\n✅ Ingestion successful!")
        print(f"   Document: {stats['document']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Images: {stats['total_images']}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        logger.exception("Ingestion error")
        return 1


if __name__ == "__main__":
    sys.exit(main())