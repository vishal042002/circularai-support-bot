"""
Image processing with GPT-4 Vision descriptions
"""

from typing import List, Dict
from pathlib import Path
from core.llm import LLMService
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Process images and generate descriptions using GPT-4 Vision"""
    
    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService()
    
    def process_images(self, images: List[Dict]) -> List[Dict]:
        """
        Process images and add descriptions
        
        Args:
            images: List of image dicts with 'path' key
        
        Returns:
            Same list with 'description' added
        """
        logger.info(f"Processing {len(images)} images with GPT-4 Vision...")
        
        for i, image in enumerate(images):
            try:
                description = self.llm.describe_image(
                    image['path'],
                    prompt=(
                        "Describe this screenshot from the CircularAI platform user interface. "
                        "Focus on: buttons, menus, navigation elements, forms, and any visible text. "
                        "Be specific about UI element locations (top-right, left sidebar, etc.). "
                        "Keep description under 200 words."
                    )
                )
                
                image['description'] = description
                logger.info(f"  [{i+1}/{len(images)}] Processed {image['filename']}")
            
            except Exception as e:
                logger.error(f"  [{i+1}/{len(images)}] Failed to process {image['filename']}: {e}")
                image['description'] = f"[Screenshot from CircularAI interface: {image['filename']}]"
        
        return images
    
    def create_image_context(self, images: List[Dict]) -> str:
        """Create context string from images for inclusion in chunks"""
        if not images:
            return ""
        
        context_parts = ["\n[Related Screenshots:]"]
        for image in images:
            description = image.get('description', '')
            if description:
                context_parts.append(f"- {description}")
        
        return "\n".join(context_parts)