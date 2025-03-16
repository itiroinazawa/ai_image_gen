#!/usr/bin/env python
"""
Video Generation Example

This script demonstrates how to use the VideoGenerationAgent to generate a video from a text prompt.
"""
import os
import sys
import time

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.video_gen_agent import VideoGenerationAgent
from utils.config import Config


def run_video_generation_example():
    """
    Run an example of video generation.

    Returns:
        str: Path to the generated video
    """
    print("Running video generation example...")

    # Initialize the agent
    config = Config()
    agent = VideoGenerationAgent(config)

    # Hardcoded parameters
    prompt = "A spaceship flying through an asteroid field with stars in the background"
    negative_prompt = "blurry, low quality, explosion, crash"
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    # num_inference_steps = 30
    num_inference_steps = 20
    guidance_scale = 7.5
    num_frames = 16
    height = 256
    width = 256
    seed = 42

    # Generate the video
    start_time = time.time()
    output_path = agent.generate_video(
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
    end_time = time.time()

    print(f"Generated video saved to: {output_path}")
    print(f"Video generation took {end_time - start_time:.2f} seconds")

    return output_path


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Run video generation example
    video_path = run_video_generation_example()

    print(
        f"Video generation example completed successfully. Video saved to: {video_path}"
    )
