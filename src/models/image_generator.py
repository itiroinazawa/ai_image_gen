"""
Image generator model for text-to-image generation.
"""

import logging
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image

from checkers.device_checker import verify_device, verify_precision
from models.utils.image_models_util import ImageModelsUtil
from utils.config import Config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Image generator for text-to-image generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the image generator.

        Args:
            config: Configuration object
        """
        self.config = config

        # Verify CUDA availability and set device accordingly
        self.device = verify_device(logger, config.device)
        self.precision = verify_precision(logger, self.device, config.precision)

        logger.info(f"Using device: {self.device}, precision: {self.precision}")

        # Cache for loaded models
        self._model_cache = {}

        self._image_models_util = ImageModelsUtil(config)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
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
            height: Height of the generated image
            width: Width of the generated image

        Returns:
            Generated image
        """
        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            f"Generating image with prompt: {prompt}, model: {model_id}, seed: {seed}"
        )

        # Load the model
        pipe = self._image_models_util._load_model_txt_2_img(model_id)

        # Load LoRA if specified
        if lora_id is not None:
            pipe = self._image_models_util._load_lora(pipe, lora_id)

        # Generate the image
        with torch.autocast(self.device.type, enabled=self.precision == "fp16"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
            )

        # Convert PIL image to NumPy array
        image_array = (
            np.array(result.images[0]).astype(np.float32) / 255.0
        )  # Normalize to [0,1]

        # Convert NumPy array to Tensor
        image_tensor = torch.from_numpy(image_array)

        # Handle potential NaN/infinity issues
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure values are in [0,1]

        # Convert back to NumPy array and scale to 255
        image_array = (image_tensor * 255).byte().cpu().numpy()

        # Convert NumPy array back to PIL image
        image = Image.fromarray(image_array)

        return image
