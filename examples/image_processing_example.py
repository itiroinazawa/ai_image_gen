#!/usr/bin/env python
"""
Image Processing Example

This script demonstrates how to use the ImageChangerAgent to process an image from a URL using a text prompt.
"""
import os
import sys
from PIL import Image
from io import BytesIO

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.image_changer_agent import ImageChangerAgent
from src.utils.config import Config


def run_image_processing_example():
    """
    Run an example of image processing from a URL.
    
    Returns:
        str: Path to the processed image
    """
    print("Running image processing example...")
    
    # Initialize the agent
    config = Config()
    agent = ImageChangerAgent(config)
    
    # Hardcoded parameters
    image_url = "https://images.unsplash.com/photo-1611597617014-9970403724e9?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    prompt = "Transform this landscape into a winter wonderland with snow-covered mountains"
    negative_prompt = "summer, green, warm colors"
    model_id = "runwayml/stable-diffusion-v1-5"
    #num_inference_steps = 30
    num_inference_steps = 10
    guidance_scale = 7.5
    strength = 0.8
    seed = 42
    
    # Process the image
    output_path = agent.process_image_url(
        image_url=image_url,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
    )
    
    print(f"Processed image saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Run image processing example
    processed_image_path = run_image_processing_example()
    
    print(f"Image processing example completed successfully. Image saved to: {processed_image_path}")
