"""
Hierarchical document chunking that respects structure
"""

from typing import List, Dict, Tuple
import re
import unicodedata
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HierarchicalChunker:
    """
    Chunking strategy that preserves document hierarchy and procedural sequences
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        respect_sections: bool = None,
        preserve_steps: bool = None
    ):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.respect_sections = respect_sections if respect_sections is not None else config.chunking.respect_sections
        self.preserve_steps = preserve_steps if preserve_steps is not None else config.chunking.preserve_steps
        
        logger.info(f"Chunker: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk document with hierarchy awareness
        
        Args:
            document: Dict with 'sections' list containing hierarchical structure
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for section in document.get('sections', []):
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_section(self, section: Dict) -> List[Dict]:
        """Chunk a single section"""
        chunks = []
        
        section_title = section.get('title', '')
        section_level = section.get('level', 1)
        section_text = section.get('text', '')
        
        # If section is small enough, keep it intact
        if len(section_text) <= self.chunk_size:
            chunks.append({
                'text': section_text,
                'metadata': {
                    'section_title': section_title,
                    'section_level': section_level,
                    'chunk_type': 'complete_section',
                    'has_subsections': len(section.get('subsections', [])) > 0
                }
            })
        else:
            # Split large sections
            if self.preserve_steps and self._is_procedural(section_text):
                # Special handling for step-by-step content
                step_chunks = self._chunk_procedural(section_text)
                for i, chunk_text in enumerate(step_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'section_title': section_title,
                            'section_level': section_level,
                            'chunk_type': 'procedural',
                            'chunk_index': i,
                            'has_steps': True
                        }
                    })
            else:
                # Standard recursive chunking
                text_chunks = self._recursive_split(section_text)
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'section_title': section_title,
                            'section_level': section_level,
                            'chunk_type': 'standard',
                            'chunk_index': i
                        }
                    })
        
        # Process subsections recursively
        for subsection in section.get('subsections', []):
            subsection_chunks = self._chunk_section(subsection)
            chunks.extend(subsection_chunks)
        
        return chunks
    
    def _is_procedural(self, text: str) -> bool:
        """Detect if text contains step-by-step instructions"""
        procedural_patterns = [
            r'\d+\.',  # Numbered lists
            r'step \d+',  # "Step 1", "Step 2"
            r'first,.*then,.*finally',  # Sequential words
            r'click.*then.*select',  # UI instructions
        ]
        
        text_lower = text.lower()
        for pattern in procedural_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _chunk_procedural(self, text: str) -> List[str]:
        """
        Chunk procedural text while keeping step sequences together
        """
        chunks = []
        
        # Split by numbered steps or major breaks
        # Pattern: "1. ", "2. ", etc.
        steps = re.split(r'(\d+\.\s+)', text)
        
        # Reconstruct steps
        current_chunk = ""
        
        for i in range(1, len(steps), 2):  # Skip first empty and iterate pairs
            if i + 1 < len(steps):
                step_marker = steps[i]
                step_content = steps[i + 1]
                step_text = step_marker + step_content
                
                # If adding this step exceeds size, save current chunk
                if len(current_chunk) + len(step_text) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap (include previous step)
                    if chunks:
                        # Get last step from previous chunk for context
                        prev_steps = re.split(r'(\d+\.\s+)', current_chunk)
                        if len(prev_steps) >= 3:
                            overlap_text = prev_steps[-2] + prev_steps[-1]
                            current_chunk = overlap_text + "\n\n" + step_text
                        else:
                            current_chunk = step_text
                    else:
                        current_chunk = step_text
                else:
                    current_chunk += "\n\n" + step_text if current_chunk else step_text
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text by paragraphs, then sentences, then characters
        """
        chunks = []
        
        # Level 1: Split by double newline (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = " ".join(overlap_words) + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Level 2: If any chunk is still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""
                
                for sent in sentences:
                    if len(sub_chunk) + len(sent) > self.chunk_size and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                    else:
                        sub_chunk += " " + sent if sub_chunk else sent
                
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
        
        return final_chunks
    
    def _sanitize_id(self, text: str) -> str:
        """
        Sanitize text to create valid Pinecone IDs (ASCII only)
        
        Replaces non-ASCII characters and spaces with underscores
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters, replace with underscore
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Replace spaces and special characters with underscores
        text = re.sub(r'[^\w\-]', '_', text)
        
        # Remove consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        # Remove leading/trailing underscores
        text = text.strip('_')
        
        # Limit length to 512 characters (Pinecone limit)
        if len(text) > 500:
            text = text[:500]
        
        return text
    
    def chunk_simple(self, text: str, doc_id: str, metadata: Dict = None) -> List[Dict]:
        """
        Simple chunking without hierarchy (fallback method)
        
        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Additional metadata
        
        Returns:
            List of chunks with metadata
        """
        chunks = self._recursive_split(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'doc_id': doc_id,
                'chunk_index': i,
                'chunk_type': 'simple',
                **(metadata or {})
            }
            
            # Sanitize doc_id to only ASCII characters
            safe_doc_id = self._sanitize_id(doc_id)
            
            result.append({
                'id': f"{safe_doc_id}_chunk_{i}",
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result