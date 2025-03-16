#!/usr/bin/env python3
"""
RunPod serverless handler for the AI Image & Video Generation Agent.

This script provides a serverless interface for the agent, allowing it to be
deployed on RunPod's serverless infrastructure.
"""
# Add the project root to the Python path to allow for absolute imports
import sys
import os

# Get the directory of this file and add its parent (project root) to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
import base64
from io import BytesIO
from PIL import Image

import runpod
from runpod.serverless.utils import rp_download, upload_file_to_bucket, rp_cleanup
from runpod.serverless.modules.rp_logger import RunPodLogger

# Import the agents and config
from agent.image_gen_agent import ImageGenerationAgent
from agent.video_gen_agent import VideoGenerationAgent
from agent.image_changer_agent import ImageChangerAgent
from utils.config import Config
import boto3

# Initialize logger
logger = RunPodLogger()

# Global agent instances
image_agent = None
video_agent = None
image_changer_agent = None


def initialize_agents():
    """
    Initialize all AI Generation Agents.
    """
    global image_agent, video_agent, image_changer_agent
    
    try:
        logger.info("Initializing AI Generation Agents...")
        config = Config()
        
        # Initialize all agents if they don't exist
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
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            initialize_agents()
        else:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error initializing agents: {str(e)}\nStacktrace:\n{error_traceback}")
            raise e
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error initializing agents: {str(e)}\nStacktrace:\n{error_traceback}")
        raise e

def save_file(output_path):
    bucket_url = os.environ.get("BUCKET_ENDPOINT_URL")
    bucket_access_key = os.environ.get("BUCKET_ACCESS_KEY_ID")
    bucket_secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    
    bucket_creds = {
        'endpointUrl': bucket_url,
        'accessId': bucket_access_key,
        'accessSecret': bucket_secret_key
    }

    logger.info("Saving File...")

    # Upload the output file to RunPod storage if available
    if output_path:
        if os.path.exists(output_path):
            logger.info(f"File {output_path} exists, uploading to storage...")
            # Upload to RunPod storage and get a public URL
            filename = os.path.basename(output_path)            
            
            presigned_url = upload_file_to_bucket(filename, output_path, bucket_creds)
            return presigned_url
        else:
            logger.warn(f"File {output_path} does not exist, skipping upload.")
            return None
    else:
        logger.warn("No output path provided, skipping file upload.")
        return None

def save_file_fallback(output_path):
    filename = os.path.basename(output_path)
    bucket_name = os.getenv("S3_BUCKET_NAME")

    s3_client = boto3.client('s3')
    s3_client.upload_file(output_path, bucket_name, filename)
    
    logger.info(f"File {filename} uploaded to S3 bucket: {bucket_name}")

    return f"https://{bucket_name}.s3.amazonaws.com/{filename}"


def handler(event):
    """
    RunPod serverless handler function.
    
    This function receives requests from the RunPod serverless infrastructure
    and processes them using the appropriate AI Generation Agent.
    
    Args:
        event: Event data from RunPod
        
    Returns:
        Dictionary with the processing result
    """
    global image_agent, video_agent, image_changer_agent
    
    # Initialize the agents if not already initialized
    if image_agent is None or video_agent is None or image_changer_agent is None:
        initialize_agents()
    
    try:
        # Start timing
        start_time = time.time()
        
        # Extract request data
        input_data = event.get("input", {})
        method = input_data.get("method", "")
        
        # Common parameters
        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", None)
        model_id = input_data.get("model_id", "runwayml/stable-diffusion-v1-5")
        lora_id = input_data.get("lora_id", None)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        strength = input_data.get("strength", 0.8)
        seed = input_data.get("seed", None)
        
        # Video-specific parameters
        num_frames = input_data.get("num_frames", 16)
        height = input_data.get("height", 512)
        width = input_data.get("width", 512)
        
        result = None
        output_path = None

        # Process the request based on the method
        if method == "generate_image":
            logger.info(f"Generating image with prompt: {prompt}")
            output_path = image_agent.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                lora_id=lora_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed
            )
            result = {"output_path": output_path}
            
        elif method == "process_image":
            # Handle image data (base64 or URL)
            image_data = input_data.get("image", None)
            if image_data:
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    # Handle base64 encoded image
                    base64_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_bytes))
                else:
                    # Download from temporary URL if provided by RunPod
                    image_path = rp_download(image_data)
                    image = Image.open(image_path)
                
                logger.info(f"Processing image with prompt: {prompt}")
                output_path = image_changer_agent.process_image(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                    lora_id=lora_id,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed
                )
                result = {"output_path": output_path}
                
                # Clean up temporary files
                if isinstance(image_data, str) and not image_data.startswith("data:image"):
                    rp_cleanup(image_path)
            else:
                return {"error": "No image data provided"}
            
        elif method == "process_image_url":
            image_url = input_data.get("image_url", None)
            if image_url:
                logger.info(f"Processing image from URL: {image_url}")
                output_path = image_changer_agent.process_image_url(
                    image_url=image_url,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                    lora_id=lora_id,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed
                )
                result = {"output_path": output_path}
            else:
                return {"error": "No image URL provided"}
            
        elif method == "generate_video":
            logger.info(f"Generating video with prompt: {prompt}")
            output_path = video_agent.generate_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=seed
            )
            result = {"output_path": output_path}
            
        elif method == "list_loras":
            logger.info("Listing available LoRAs")
            loras = image_agent.list_loras()
            result = {"loras": loras}
            
        elif method == "list_models":
            logger.info("Listing available models")
            image_models = image_agent.list_models()
            video_models = video_agent.list_models()
            result = {"models": image_models + video_models}
            
        else:
            return {"error": f"Unknown method: {method}"}

        if output_path:
            logger.info(f"Output path: {output_path}")
            #full_path = os.path.join("output", output_path)
            full_path = output_path
            presigned_url = save_file(full_path)
            
            if presigned_url:
                result["public_url"] = presigned_url
                logger.info(f"Runpod File saved successfully: {full_path}")

            else:
                presigned_url = save_file_fallback(full_path)
                result["public_url"] = presigned_url
                logger.info(f"Boto File saved successfully: {full_path}")


        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        # Get device information
        device = "CPU" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "GPU"
        
        # Return the result
        return {
            "output": {
                "result": result,
                "processing_time": f"{elapsed_time:.2f} seconds",
                "device_used": device
            }
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing request: {str(e)}\nStacktrace:\n{error_traceback}")
        return {
            "error": str(e),
            "traceback": error_traceback
        }


# Initialize the agents
logger.info("Starting agents initialization...")
initialize_agents()

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
