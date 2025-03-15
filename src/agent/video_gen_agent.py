"""
AI Generation Agent for image and video processing and generation.
"""
import os
import uuid
import logging
from typing import Dict, List, Optional, Union, Any
import random
from src.models.video_generator import VideoGenerator
from src.utils.config import Config

logger = logging.getLogger(__name__)


class VideoGenerationAgent:
    """
    Video Generation Agent for video processing and generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the Video Generation Agent.

        Args:
            config: Configuration object
        """
        self.config = config
        self.video_generator = VideoGenerator(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)

    def generate_video(
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
    ) -> str:
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

        Returns:
            Path to the generated video
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        logger.info(f"Generating video with prompt: {prompt}")
        
        # Generate the video
        video = self.video_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        
        # Save the video
        output_path = self._save_video(video)
        
        return output_path

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models.

        Returns:
            List of available models with their IDs and names
        """
        # Combine models from image and video generators
        video_models = self.video_generator.list_models()
        
        return video_models

    def _save_video(self, video: Any) -> str:
        """
        Save a video to the output directory.

        Args:
            video: Video to save (format depends on the video generator)

        Returns:
            Path to the saved video
        """
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(self.config.output_dir, filename)
        
        # Save the video (implementation depends on the video format)
        self.video_generator.save_video(video, output_path)
        
        return filename
