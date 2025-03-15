"""
AI Generation Agent for image and video processing and generation.
"""
import os
import uuid
import logging
from typing import Dict, List, Optional, Union, Any
import requests
from io import BytesIO
import random
import torch
from fastapi import UploadFile
from PIL import Image

from src.models.image_generator import ImageGenerator
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ImageGenerationAgent:
    """
    AI Generation Agent for image generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the Image Generation Agent.

        Args:
            config: Configuration object
        """
        self.config = config
        self.image_generator = ImageGenerator(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative text prompt for image generation
            model_id: Model ID to use for generation
            lora_id: LoRA model ID to use for generation
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed for reproducibility

        Returns:
            Path to the generated image
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Generate the image
        image = self.image_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )
        
        # Save the image
        output_path = self._save_image(image)
        
        return output_path

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models.

        Returns:
            List of available models with their IDs and names
        """
        return self.image_generator.list_models()

    def list_loras(self) -> List[Dict[str, str]]:
        """
        List available LoRA models.

        Returns:
            List of available LoRA models with their IDs and names
        """
        return self.image_generator.list_loras()

    def _save_image(self, image: Image.Image) -> str:
        """
        Save an image to the output directory.

        Args:
            image: PIL Image to save

        Returns:
            Path to the saved image
        """
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join(self.config.output_dir, filename)
        
        # Save the image
        image.save(output_path)
        
        return filename
