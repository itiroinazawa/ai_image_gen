"""
AI Generation Agent for image and video processing and generation.
"""
import sys
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

# Import from relative paths
from ..models.image_processor import ImageProcessor
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ImageChangerAgent:
    """
    Image Changer Agent for image processing.
    """

    def __init__(self, config: Config):
        """
        Initialize the AI Generation Agent.

        Args:
            config: Configuration object
        """
        self.config = config
        self.image_processor = ImageProcessor(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)

    def process_image(
        self,
        image: UploadFile,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> str:
        """
        Process an uploaded image using a text prompt.

        Args:
            image: Uploaded image file
            prompt: Text prompt for image processing
            negative_prompt: Negative text prompt for image processing
            model_id: Model ID to use for processing
            lora_id: LoRA model ID to use for processing
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Strength of the processing (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Path to the processed image
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        logger.info(f"Processing image with prompt: {prompt}")
        
        # Read the image
        contents = image.file.read()
        input_image = Image.open(BytesIO(contents))
        
        # Process the image
        processed_image = self.image_processor.process(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        
        # Save the image
        output_path = self._save_image(processed_image)
        
        return output_path

    def process_image_url(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> str:
        """
        Process an image from a URL using a text prompt.

        Args:
            image_url: URL of the image to process
            prompt: Text prompt for image processing
            negative_prompt: Negative text prompt for image processing
            model_id: Model ID to use for processing
            lora_id: LoRA model ID to use for processing
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Strength of the processing (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Path to the processed image
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        logger.info(f"Processing image from URL: {image_url} with prompt: {prompt}")
        
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        input_image = Image.open(BytesIO(response.content))
        
        # Process the image
        processed_image = self.image_processor.process(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        
        # Save the image
        output_path = self._save_image(processed_image)
        
        return output_path


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
