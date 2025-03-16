"""
Video generator model for text-to-video generation.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    TextToVideoSDPipeline,
    TextToVideoZeroPipeline,
)
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image

from checkers.device_checker import verify_device, verify_precision
from utils.config import Config

logger = logging.getLogger(__name__)


class VideoGenerator:
    """
    Video generator for text-to-video generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the video generator.

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
        model_id: str = "damo-vilab/text-to-video-ms-1.7b",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        seed: Optional[int] = None,
        fps: int = 8,
    ) -> Any:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative text prompt for video generation
            model_id: Model ID to use for generation
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            num_frames: Number of frames to generate
            height: Height of the video
            width: Width of the video
            seed: Random seed for reproducibility
            fps: Frames per second for the output video

        Returns:
            Generated video frames
        """
        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            f"Generating video with prompt: {prompt}, model: {model_id}, seed: {seed}"
        )

        # Load the model
        pipe = self._load_model(model_id)

        # Generate the video
        with torch.autocast(self.device.type, enabled=self.precision == "fp16"):
            if model_id == "damo-vilab/text-to-video-ms-1.7b":
                # ModelScope model has a different API
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    generator=generator,
                )
                frames = result.frames[0]
            elif "zeroscope" in model_id:
                # ZeroScope model
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    generator=generator,
                )
                frames = result.frames[0]
            else:
                # Generic text-to-video model
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    generator=generator,
                )
                frames = result.frames[0]

        # Return the generated video frames along with metadata
        return {
            "frames": frames,
            "fps": fps,
            "prompt": prompt,
        }

    def save_video(self, video: Any, output_path: str) -> None:
        """
        Save a video to a file.

        Args:
            video: Video to save (format depends on the video generator)
            output_path: Path to save the video to
        """
        frames = video["frames"]
        fps = video.get("fps", 8)

        # Convert frames to PIL images if they're not already
        if not isinstance(frames[0], Image.Image):
            frames = [Image.fromarray(frame) for frame in frames]

        # Create a temporary directory for the frames
        temp_dir = os.path.join(self.config.cache_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Save the frames as images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # Create a video from the frames
        clip = ImageSequenceClip(frame_paths, fps=fps)
        clip.write_videofile(output_path, codec="libx264", fps=fps)

        # Clean up the temporary files
        for frame_path in frame_paths:
            os.remove(frame_path)

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
