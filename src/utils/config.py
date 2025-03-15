"""
Configuration utilities for the AI Image & Video Generation Agent.
"""
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration for the AI Image & Video Generation Agent.
    """

    # General settings
    output_dir: str = Field(default="output")
    cache_dir: str = Field(default="cache")
    
    # Model settings
    default_image_model: str = Field(default="runwayml/stable-diffusion-v1-5")
    default_video_model: str = Field(default="damo-vilab/text-to-video-ms-1.7b")
    
    # Generation settings
    default_num_inference_steps: int = Field(default=50)
    default_guidance_scale: float = Field(default=7.5)
    default_strength: float = Field(default=0.8)
    
    # Hardware settings
    device: str = Field(default="cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu")
    precision: str = Field(default="fp16" if os.environ.get("USE_FP16", "1") == "1" else "fp32")
    
    # API settings
    huggingface_token: Optional[str] = Field(default=os.environ.get("HUGGINGFACE_TOKEN"))
    
    # Available models
    image_models: Dict[str, Dict[str, Any]] = Field(
        default={
            "runwayml/stable-diffusion-v1-5": {
                "name": "Stable Diffusion v1.5",
                "type": "text-to-image",
            },
            "stabilityai/stable-diffusion-2-1": {
                "name": "Stable Diffusion v2.1",
                "type": "text-to-image",
            },
            "CompVis/stable-diffusion-v1-4": {
                "name": "Stable Diffusion v1.4",
                "type": "text-to-image",
            },
            "stabilityai/stable-diffusion-xl-base-1.0": {
                "name": "Stable Diffusion XL",
                "type": "text-to-image",
            },
        }
    )
    
    video_models: Dict[str, Dict[str, Any]] = Field(
        default={
            "damo-vilab/text-to-video-ms-1.7b": {
                "name": "ModelScope Text-to-Video",
                "type": "text-to-video",
            },
            "cerspense/zeroscope_v2_576w": {
                "name": "ZeroScope v2",
                "type": "text-to-video",
            },
        }
    )
    
    # Available LoRAs
    loras: Dict[str, Dict[str, Any]] = Field(
        default={
            "sayakpaul/sd-model-finetuned-lora-t4": {
                "name": "Pokémon LoRA",
                "type": "lora",
            },
            "ostris/ikea-instructions-lora": {
                "name": "IKEA Instructions LoRA",
                "type": "lora",
            },
        }
    )
    
    def __init__(self, **data):
        """
        Initialize the configuration.
        
        Args:
            **data: Configuration data
        """
        super().__init__(**data)
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
