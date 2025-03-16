#!/usr/bin/env python3
"""
Main entry point for the AI Image & Video Generation Agent.
"""
import os

# Add the project root to the Python path to allow for absolute imports
import sys

# Get the directory of this file and add its parent (project root) to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import argparse
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.image_changer_agent import ImageChangerAgent
from agent.image_gen_agent import ImageGenerationAgent
from agent.video_gen_agent import VideoGenerationAgent
from utils.config import Config

# Load environment variables
load_dotenv()

# Initialize the application
app = FastAPI(
    title="AI Image & Video Generation Agent",
    description="An AI-powered agent for image processing and generation, as well as video generation from text prompts.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize the agents
config = Config()
image_agent = ImageGenerationAgent(config)
video_agent = VideoGenerationAgent(config)
image_changer_agent = ImageChangerAgent(config)

# Mount static files directory
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Image & Video Generation Agent API"}


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration systems."""
    return {"status": "healthy"}


@app.post("/generate-image")
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    model_id: str = Form("runwayml/stable-diffusion-v1-5"),
    lora_id: str = Form(None),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    seed: int = Form(None),
):
    """Generate an image from a text prompt."""
    try:
        output_path = image_agent.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            seed=seed,
        )
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    model_id: str = Form("runwayml/stable-diffusion-v1-5"),
    lora_id: str = Form(None),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.8),
    seed: int = Form(None),
):
    """Process an uploaded image using a text prompt."""
    try:
        output_path = image_changer_agent.process_image(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-image-url")
async def process_image_url(
    image_url: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    model_id: str = Form("runwayml/stable-diffusion-v1-5"),
    lora_id: str = Form(None),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.8),
    seed: int = Form(None),
):
    """Process an image from a URL using a text prompt."""
    try:
        output_path = image_changer_agent.process_image_url(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            lora_id=lora_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-video")
async def generate_video(
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    model_id: str = Form("damo-vilab/text-to-video-ms-1.7b"),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    num_frames: int = Form(16),
    height: int = Form(256),
    width: int = Form(256),
    seed: int = Form(None),
):
    """Generate a video from a text prompt."""
    try:
        output_path = video_agent.generate_video(
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
        return {"status": "success", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    image_models = image_agent.list_models()
    video_models = video_agent.list_models()
    return {"models": image_models + video_models}


@app.get("/loras")
async def list_loras():
    """List available LoRA models."""
    return {"loras": image_agent.list_loras()}


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Get a generated output file."""
    file_path = os.path.join("output", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Image & Video Generation Agent")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
    )
