import logging
from typing import Dict, List

import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    TextToVideoSDPipeline,
    TextToVideoZeroPipeline,
)

from checkers.device_checker import verify_device, verify_precision
from utils.config import Config

logger = logging.getLogger(__name__)


class VideoModelsUtil:
    def __init__(self, config: Config):
        self.config = config
        self._model_cache = {}
        self.device = verify_device(logger, config.device)
        self.precision = verify_precision(logger, self.device, config.precision)

        logger.info(f"Using device: {self.device}, precision: {self.precision}")

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models.

        Returns:
            List of available models with their IDs and names
        """
        return [
            {"id": model_id, "name": model_info["name"], "type": model_info["type"]}
            for model_id, model_info in self.config.video_models.items()
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

        logger.info(f"Loading video model: {model_id}")

        # Load the appropriate pipeline based on the model ID
        if model_id == "damo-vilab/text-to-video-ms-1.7b":
            # ModelScope text-to-video model
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
                variant="fp16" if self.precision == "fp16" else None,
                cache_dir=self.config.cache_dir,
                token=self.config.huggingface_token,
            )
        elif "zeroscope" in model_id:
            # ZeroScope text-to-video model
            pipe = TextToVideoZeroPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
                variant="fp16" if self.precision == "fp16" else None,
                cache_dir=self.config.cache_dir,
                token=self.config.huggingface_token,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
        else:
            # Generic text-to-video model
            pipe = TextToVideoSDPipeline.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16 if self.precision == "fp16" else torch.float32
                ),
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
