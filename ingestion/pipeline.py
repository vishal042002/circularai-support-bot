"""
Complete document ingestion pipeline
"""

from typing import List, Dict
from pathlib import Path
import json
from ingestion.docx_loader import DOCXLoader
from ingestion.image_processor import ImageProcessor
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.chunking import HierarchicalChunker
from core.search import HybridSearch
from utils.logging_config import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Complete pipeline for document ingestion:
    1. Load DOCX with structure and images
    2. Process images with GPT-4 Vision
    3. Chunk documents hierarchically
    4. Generate embeddings
    5. Upload to vector store
    6. Build BM25 index
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        search_engine: HybridSearch
    ):
        self.embedder = embedding_service
        self.vector_store = vector_store
        self.search_engine = search_engine
        
        self.docx_loader = DOCXLoader()
        self.image_processor = ImageProcessor()
        self.chunker = HierarchicalChunker()
    
    def ingest_docx(
        self,
        docx_path: str,
        process_images: bool = True,
        save_processed: bool = True
    ) -> Dict:
        """
        Ingest DOCX file through complete pipeline
        
        Args:
            docx_path: Path to DOCX file
            process_images: Whether to generate image descriptions
            save_processed: Whether to save processed data
        
        Returns:
            Dict with ingestion statistics
        """
        logger.info("=" * 80)
        logger.info("STARTING DOCUMENT INGESTION PIPELINE")
        logger.info("=" * 80)

        # Clean existing vectors to prevent duplicates
        logger.info("\n[Pre-check] Checking for existing vectors...")
        existing_stats = self.vector_store.get_stats()
        existing_count = existing_stats.get('total_vector_count', 0)

        if existing_count > 0:
            logger.warning(f"  Found {existing_count} existing vectors in Pinecone")
            logger.warning("  Deleting all existing vectors to prevent duplicates...")
            self.vector_store.delete_all()
            logger.info("  Pinecone cleared successfully")
        else:
            logger.info("  Pinecone is empty - ready for ingestion")

        # Step 1: Load DOCX
        logger.info("\n[1/6] Loading DOCX file...")
        document = self.docx_loader.load(docx_path)
        logger.info(f"  Loaded: {document['metadata']['total_paragraphs']} paragraphs, "
                   f"{document['metadata']['total_images']} images")
        
        # Step 2: Process images
        if process_images and document['images']:
            logger.info(f"\n[2/6] Processing {len(document['images'])} images with GPT-4 Vision...")
            document['images'] = self.image_processor.process_images(document['images'])
        else:
            logger.info("\n[2/6] Skipping image processing")
        
        # Step 3: Chunk documents
        logger.info("\n[3/6] Chunking document hierarchically...")
        chunks = self._create_chunks(document)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Step 4: Generate embeddings
        logger.info(f"\n[4/6] Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        logger.info(f"  Generated {len(embeddings)} embeddings")
        
        # Step 5: Upload to Pinecone
        logger.info("\n[5/6] Uploading to vector store...")
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                'id': chunk['id'],
                'values': embedding,
                'metadata': {
                    'text': chunk['text'],
                    **chunk['metadata']
                }
            })
        
        self.vector_store.upsert(vectors)
        logger.info(f"  Uploaded {len(vectors)} vectors")
        
        # Step 6: Build BM25 index
        logger.info("\n[6/6] Building BM25 index...")
        self.search_engine.build_bm25_index(chunks)
        
        # Save processed data
        if save_processed:
            self._save_processed_data(document, chunks, Path(docx_path).stem)
        
        # Statistics
        stats = {
            'document': document['metadata']['filename'],
            'total_chunks': len(chunks),
            'total_images': len(document['images']),
            'images_processed': process_images,
            'sections': len(document['sections'])
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Document: {stats['document']}")
        logger.info(f"Chunks created: {stats['total_chunks']}")
        logger.info(f"Images processed: {stats['total_images']}")
        logger.info("=" * 80 + "\n")
        
        return stats
    
    def _create_chunks(self, document: Dict) -> List[Dict]:
        """Create chunks from document sections"""
        all_chunks = []
        doc_name = document['metadata']['filename']
        
        for section in document['sections']:
            # Process section and subsections
            section_chunks = self._process_section(section, doc_name)
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _process_section(self, section: Dict, doc_name: str, parent_titles: List[str] = None) -> List[Dict]:
        """Recursively process section and subsections"""
        parent_titles = parent_titles or []
        chunks = []
        
        section_title = section['title']
        section_text = section['text']
        section_level = section['level']
        section_images = section.get('images', [])
        
        # Add image context to text
        if section_images:
            image_context = self.image_processor.create_image_context(section_images)
            section_text += image_context
        
        # Chunk this section
        if section_text.strip():
            section_chunks = self.chunker.chunk_simple(
                text=section_text,
                doc_id=f"{doc_name}_{section_title.replace(' ', '_')}",
                metadata={
                    'section_h1': parent_titles[0] if len(parent_titles) > 0 else section_title,
                    'section_h2': parent_titles[1] if len(parent_titles) > 1 else (section_title if section_level == 2 else ''),
                    'section_h3': section_title if section_level == 3 else '',
                    'section_level': section_level,
                    'has_images': len(section_images) > 0,
                    'image_count': len(section_images),
                    'source_document': doc_name
                }
            )
            chunks.extend(section_chunks)
        
        # Process subsections
        for subsection in section.get('subsections', []):
            subsection_chunks = self._process_section(
                subsection,
                doc_name,
                parent_titles + [section_title]
            )
            chunks.extend(subsection_chunks)
        
        return chunks
    
    def _save_processed_data(self, document: Dict, chunks: List[Dict], doc_name: str):
        """Save processed data for reference"""
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save document structure
        structure_file = output_dir / f"{doc_name}_structure.json"
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        
        # Save chunks
        chunks_file = output_dir / f"{doc_name}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved processed data to {output_dir}")