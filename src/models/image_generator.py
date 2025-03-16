"""
Image generator model for text-to-image generation.
"""

import logging
import random
from typing import Dict, List, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import LoraLoaderMixin
from PIL import Image

from checkers.device_checker import verify_device, verify_precision
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
        pipe = self._load_model(model_id)

        # Load LoRA if specified
        if lora_id is not None:
            pipe = self._load_lora(pipe, lora_id)

        # Generate the image
        with torch.autocast(self.device.type, enabled=self.precision == "fp16"):
            if "stable-diffusion-xl" in model_id:
                # SDXL has a different API
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )

        # Return the generated image
        return result.images[0]

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models.

        Returns:
            List of available models with their IDs and names
        """
        return [
            {"id": model_id, "name": model_info["name"], "type": model_info["type"]}
            for model_id, model_info in self.config.image_models.items()
        ]

    def list_loras(self) -> List[Dict[str, str]]:
        """
        List available LoRA models.

        Returns:
            List of available LoRA models with their IDs and names
        """
        return [
            {"id": lora_id, "name": lora_info["name"], "type": lora_info["type"]}
            for lora_id, lora_info in self.config.loras.items()
        ]

    def _load_model(self, model_id: str) -> DiffusionPipeline:
        """
        Load a model.

        Args:
            model_id: Model ID to load

        Returns:
            Loaded model
        """
        # Check if the model is already loaded
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        logger.info(f"Loading model: {model_id}")

        # Load the appropriate pipeline based on the model ID
        if "stable-diffusion-xl" in model_id:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
                use_safetensors=True,
                variant="fp16" if self.precision == "fp16" else None,
                cache_dir=self.config.cache_dir,
                token=self.config.huggingface_token,
            )
        elif "FLUX" in model_id:
            pipe = FluxPipeline.from_pretrained(
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
            pipe = StableDiffusionPipeline.from_pretrained(
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
        self._model_cache[model_id] = pipe

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
