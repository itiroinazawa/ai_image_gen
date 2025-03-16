#!/usr/bin/env python
"""
Image Generation Example

This script demonstrates how to use the ImageGenerationAgent to generate an image from a text prompt.
"""
import os
import sys

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from agent.image_gen_agent import ImageGenerationAgent
from utils.config import Config


def run_image_generation_example():
    """
    Run an example of image generation.

    Returns:
        str: Path to the generated image
    """
    print("Running image generation example...")

    # Initialize the agent
    config = Config()
    agent = ImageGenerationAgent(config)

    # Hardcoded parameters
    prompt = "A futuristic cityscape with flying cars and neon lights at night"
    negative_prompt = "daylight, sun, bright, blurry, low quality"
    model_id = "runwayml/stable-diffusion-v1-5"

    # num_inference_steps = 30
    num_inference_steps = 10
    guidance_scale = 7.5
    height = 512
    width = 512
    seed = 42

    # Generate the image
    output_path = agent.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
    )

    print(f"Generated image saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Run image generation example
    image_path = run_image_generation_example()

    # Upload image to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(image_path, "inz-runpod-bucket", "generated_image.png")

    print(
        f"Image generation example completed successfully. Image saved to: {image_path}"
    )
