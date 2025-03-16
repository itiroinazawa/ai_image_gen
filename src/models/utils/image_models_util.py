import logging
from typing import Dict, List

import torch
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from checkers.device_checker import verify_device, verify_precision
from utils.config import Config

logger = logging.getLogger(__name__)


class ImageModelsUtil:
    def __init__(self, config: Config):
        self.config = config
        self._model_cache = {}
        self.device = verify_device(logger, config.device)
        self.precision = verify_precision(logger, self.device, config.precision)

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

    def _load_model_img_2_img(self, model_id: str) -> DiffusionPipeline:
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
