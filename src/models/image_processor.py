"""
Image processor model for image-to-image generation.
"""

import logging
import random
from typing import Optional

import torch
from PIL import Image

from checkers.device_checker import verify_device, verify_precision
from models.utils.image_models_util import ImageModelsUtil
from utils.config import Config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processor for image-to-image generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the image processor.

        Args:
            config: Configuration object
        """
        self.config = config

        # Verify CUDA availability and set device accordingly
        self.device = verify_device(logger, config.device)
        self.precision = verify_precision(logger, self.device, config.precision)

        logger.info(f"Using device: {self.device}, precision: {self.precision}")

        self._image_models_util = ImageModelsUtil(config)

    def process(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_id: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Process an image using a text prompt.

        Args:
            image: Input image to process
            prompt: Text prompt for image processing
            negative_prompt: Negative text prompt for image processing
            model_id: Model ID to use for processing
            lora_id: LoRA model ID to use for processing
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Strength of the processing (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Processed image
        """
        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            f"Processing image with prompt: {prompt}, model: {model_id}, seed: {seed}"
        )

        # Load the model
        pipe = self._image_models_util._load_model_img_2_img(model_id)

        # Load LoRA if specified
        if lora_id is not None:
            pipe = self._image_models_util._load_lora(pipe, lora_id)

        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image
        with torch.autocast(self.device.type, enabled=self.precision == "fp16"):
            if "stable-diffusion-xl" in model_id:
                # SDXL has a different API
                result = pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                )
            else:
                result = pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                )

        # Return the processed image
        return result.images[0]
