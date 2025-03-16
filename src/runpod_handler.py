#!/usr/bin/env python3
"""
RunPod serverless handler for the AI Image & Video Generation Agent.

This script provides a serverless interface for the agent, allowing it to be
deployed on RunPod's serverless infrastructure.
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import sys
import time
import base64
import traceback
from io import BytesIO

# =============================================================================
# Third-Party Imports
# =============================================================================
import boto3
import runpod
from PIL import Image
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils import rp_cleanup, rp_download, upload_file_to_bucket

# =============================================================================
# Local Imports
# =============================================================================
from agent.image_changer_agent import ImageChangerAgent
from agent.image_gen_agent import ImageGenerationAgent
from agent.video_gen_agent import VideoGenerationAgent
from utils.config import Config
from utils.file_uploader import save_file

# =============================================================================
# Global Variables and Logger Initialization
# =============================================================================
logger = RunPodLogger()

# Global agent instances
image_agent = None
video_agent = None
image_changer_agent = None

# =============================================================================
# Agent Initialization
# =============================================================================
def initialize_agents():
    """
    Initialize all AI Generation Agents.
    """
    global image_agent, video_agent, image_changer_agent

    try:
        logger.info("Initializing AI Generation Agents...")
        config = Config()

        if image_agent is None:
            logger.info("Initializing Image Generation Agent...")
            image_agent = ImageGenerationAgent(config)

        if video_agent is None:
            logger.info("Initializing Video Generation Agent...")
            video_agent = VideoGenerationAgent(config)

        if image_changer_agent is None:
            logger.info("Initializing Image Changer Agent...")
            image_changer_agent = ImageChangerAgent(config)

        logger.info("All AI Generation Agents initialized successfully")
    except RuntimeError as e:
        if "HIP error" in str(e) or "CUDA error" in str(e):
            logger.warn("GPU error detected, falling back to CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            initialize_agents()
        else:
            error_traceback = traceback.format_exc()
            logger.error(f"Error initializing agents: {e}\nStacktrace:\n{error_traceback}")
            raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error initializing agents: {e}\nStacktrace:\n{error_traceback}")
        raise e


# =============================================================================
# Request Processing Functions
# =============================================================================
def process_generate_image(input_data):
    common_params = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt"),
        "model_id": input_data.get("model_id", "runwayml/stable-diffusion-v1-5"),
        "lora_id": input_data.get("lora_id"),
        "num_inference_steps": input_data.get("num_inference_steps", 50),
        "guidance_scale": input_data.get("guidance_scale", 7.5),
        "seed": input_data.get("seed"),
    }
    dimensions = {
        "height": input_data.get("height", 512),
        "width": input_data.get("width", 512),
    }
    logger.info(f"Generating image with prompt: {common_params['prompt']}")
    output_path = image_agent.generate_image(**common_params, **dimensions)
    return {"output_path": output_path}, output_path


def process_generate_video(input_data):
    common_params = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt"),
        "model_id": input_data.get("model_id", "runwayml/stable-diffusion-v1-5"),
        "num_inference_steps": input_data.get("num_inference_steps", 50),
        "guidance_scale": input_data.get("guidance_scale", 7.5),
        "seed": input_data.get("seed"),
    }
    video_params = {
        "num_frames": input_data.get("num_frames", 16),
        "height": input_data.get("height", 512),
        "width": input_data.get("width", 512),
    }
    logger.info(f"Generating video with prompt: {common_params['prompt']}")
    output_path = video_agent.generate_video(**common_params, **video_params)
    return {"output_path": output_path}, output_path


def process_image(input_data):
    common_params = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt"),
        "model_id": input_data.get("model_id", "runwayml/stable-diffusion-v1-5"),
        "lora_id": input_data.get("lora_id"),
        "num_inference_steps": input_data.get("num_inference_steps", 50),
        "guidance_scale": input_data.get("guidance_scale", 7.5),
        "strength": input_data.get("strength", 0.8),
        "seed": input_data.get("seed"),
    }
    image_data = input_data.get("image")
    if not image_data:
        return {"error": "No image data provided"}, None

    # Determine if image data is base64 encoded or a URL
    if isinstance(image_data, str) and image_data.startswith("data:image"):
        base64_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        image_downloaded = False
    else:
        image_path = rp_download(image_data)
        image = Image.open(image_path)
        image_downloaded = True

    logger.info(f"Processing image with prompt: {common_params['prompt']}")
    output_path = image_changer_agent.process_image(image=image, **common_params)

    if image_downloaded:
        rp_cleanup(image_path)
    return {"output_path": output_path}, output_path


def process_image_url(input_data):
    image_url = input_data.get("image_url")
    if not image_url:
        return {"error": "No image URL provided"}, None

    common_params = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt"),
        "model_id": input_data.get("model_id", "runwayml/stable-diffusion-v1-5"),
        "lora_id": input_data.get("lora_id"),
        "num_inference_steps": input_data.get("num_inference_steps", 50),
        "guidance_scale": input_data.get("guidance_scale", 7.5),
        "strength": input_data.get("strength", 0.8),
        "seed": input_data.get("seed"),
    }
    logger.info(f"Processing image from URL: {image_url}")
    output_path = image_changer_agent.process_image_url(image_url=image_url, **common_params)
    return {"output_path": output_path}, output_path


def list_loras(_input_data):
    logger.info("Listing available LoRAs")
    loras = image_agent.list_loras()
    return {"loras": loras}, None


def list_models(_input_data):
    logger.info("Listing available models")
    image_models = image_agent.list_models()
    video_models = video_agent.list_models()
    return {"models": image_models + video_models}, None


# =============================================================================
# Main Handler Function
# =============================================================================
def handler(event):
    """
    RunPod serverless handler function.

    Processes the incoming event by dispatching to the appropriate function
    based on the 'method' specified in the input data.
    """
    global image_agent, video_agent, image_changer_agent

    if image_agent is None or video_agent is None or image_changer_agent is None:
        initialize_agents()

    try:
        start_time = time.time()
        input_data = event.get("input", {})
        method = input_data.get("method", "")

        # Dispatch table mapping methods to their handler functions
        dispatch = {
            "generate_image": process_generate_image,
            "process_image": process_image,
            "process_image_url": process_image_url,
            "generate_video": process_generate_video,
            "list_loras": list_loras,
            "list_models": list_models,
        }

        if method not in dispatch:
            return {"error": f"Unknown method: {method}"}

        result, output_path = dispatch[method](input_data)
        
        if output_path:
            presigned_url = save_file(logger, output_path)
            result["public_url"] = presigned_url

        elapsed_time = time.time() - start_time
        device = "CPU" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "GPU"

        return {
            "output": {
                "result": result,
                "processing_time": f"{elapsed_time:.2f} seconds",
                "device_used": device,
            }
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing request: {e}\nStacktrace:\n{error_traceback}")
        return {"error": str(e), "traceback": error_traceback}


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    logger.info("Starting agents initialization...")
    initialize_agents()
    runpod.serverless.start({"handler": handler})
