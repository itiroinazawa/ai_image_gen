"""
Image processor model for image-to-image generation.
"""

import logging
import random
from typing import Optional

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.loaders import LoraLoaderMixin
from PIL import Image

from checkers.device_checker import verify_device, verify_precision
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

        # Cache for loaded models
        self._model_cache = {}

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
        pipe = self._load_model(model_id)

        # Load LoRA if specified
        if lora_id is not None:
            pipe = self._load_lora(pipe, lora_id)

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

    def _load_model(self, model_id: str) -> DiffusionPipeline:
        """
        Load a model.

        Args:
            model_id: Model ID to load

        Returns:
            Loaded model
        """
        # Check if the model is already loaded
        cache_key = f"img2img_{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        logger.info(f"Loading img2img model: {model_id}")

        # Load the appropriate pipeline based on the model ID
        if "stable-diffusion-xl" in model_id:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
                use_safetensors=True,
                variant="fp16" if self.precision == "fp16" else None,
                cache_dir=self.config.cache_dir,
                token=self.config.huggingface_token,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
                use_safetensors=True,
                variant="fp16" if self.precision == "fp16" else None,
                cache_dir=self.config.cache_dir,
                token=self.config.huggingface_token,
            )

        # Move the model to the device
        pipe = pipe.to(self.device)

        # Enable memory optimization if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        # Cache the model
        self._model_cache[cache_key] = pipe

        return pipe

    def _load_lora(self, pipe: DiffusionPipeline, lora_id: str) -> DiffusionPipeline:
        """
        Load a LoRA model.

        Args:
            pipe: Diffusion pipeline to load the LoRA into
            lora_id: LoRA model ID to load

        Returns:
            Diffusion pipeline with LoRA loaded
        """
        logger.info(f"Loading LoRA: {lora_id}")

        # Check if the pipeline supports LoRA
        if not isinstance(pipe, LoraLoaderMixin):
            logger.warning(f"Pipeline {type(pipe)} does not support LoRA")
            return pipe

        # Load the LoRA weights
        pipe.load_lora_weights(
            lora_id,
            cache_dir=self.config.cache_dir,
            token=self.config.huggingface_token,
        )

        return pipe
