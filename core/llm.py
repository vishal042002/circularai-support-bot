"""
Azure OpenAI service (Text generation + Vision)
"""

from typing import List, Dict, Optional
from openai import AzureOpenAI
import base64
from pathlib import Path
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMService:
    """Azure OpenAI service for text and vision tasks"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=config.azure.api_key,
            api_version=config.azure.api_version,
            azure_endpoint=config.azure.endpoint
        )
        self.deployment = config.azure.deployment_name
        self.vision_deployment = config.azure.vision_deployment_name
        
        logger.info(f"LLM: {self.deployment}, Vision: {self.vision_deployment}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800
    ) -> str:
        """Generate text completion"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def describe_image(
        self,
        image_path: str,
        prompt: str = "Describe this screenshot from a user interface in detail. Focus on UI elements, buttons, menus, and navigation."
    ) -> str:
        """Generate image description using GPT-4 Vision"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/png')
            
            # Call vision API
            response = self.client.chat.completions.create(
                model=self.vision_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
            logger.debug(f"Generated description for {Path(image_path).name}")
            return description
        
        except Exception as e:
            logger.error(f"Vision API error for {image_path}: {e}")
            return f"[Image: {Path(image_path).name}]"
    
    def test_connection(self) -> bool:
        """Test LLM connection"""
        try:
            response = self.generate([
                {"role": "user", "content": "Say 'Hello'"}
            ], max_tokens=10)
            
            if response:
                logger.info("✅ LLM test successful")
                return True
            return False
        
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            return False