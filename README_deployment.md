# Deployment Instructions

This document provides instructions for deploying the AI Image & Video Generation Agent on RunPod.

## RunPod Deployment

RunPod is a cloud platform that provides GPU instances for AI workloads. It's an ideal platform for deploying our AI Image & Video Generation Agent due to its serverless capabilities and GPU support.

### Prerequisites

1. A RunPod account (sign up at [runpod.io](https://www.runpod.io/))
2. Docker Hub account (optional, for custom Docker images)

### Deployment Options

#### Option 1: Using RunPod Serverless

1. Log in to your RunPod account
2. Navigate to the Serverless section
3. Create a new template with the following settings:
   - Container Image: `your-dockerhub-username/ai-image-gen:latest` (after pushing your image to Docker Hub)
   - Container Disk: `10 GB`
   - Volume Disk: `20 GB`
   - Environment Variables:
     - `HUGGINGFACE_TOKEN`: Your Hugging Face API token (if needed)
   - Ports: `8000`

4. Create a new endpoint using the template you just created
5. Once the endpoint is created, you can access the API at the provided URL

#### Option 2: Using RunPod Cloud GPU

1. Log in to your RunPod account
2. Select a GPU instance (recommended: at least 16GB VRAM)
3. Choose "Docker" as the template
4. In the advanced options, set the Docker image to `your-dockerhub-username/ai-image-gen:latest`
5. Add the following environment variables:
   - `HUGGINGFACE_TOKEN`: Your Hugging Face API token (if needed)
6. Start the pod
7. Once the pod is running, you can access the API at the provided URL

### Building and Pushing the Docker Image

Before deploying to RunPod, you need to build and push your Docker image to a registry like Docker Hub. The project now uses Poetry for dependency management, which is already configured in the Dockerfiles:

```bash
# Navigate to the project root directory
cd /path/to/ai_image_gen

# Build the Docker image
docker build -t your-dockerhub-username/ai-image-gen:latest -f deployment/Dockerfile .

# Login to Docker Hub
docker login

# Push the image to Docker Hub
docker push your-dockerhub-username/ai-image-gen:latest
```

For GPU-optimized deployment, use the GPU Dockerfile:

```bash
docker build -t your-dockerhub-username/ai-image-gen:gpu -f deployment/Dockerfile.gpu .
docker push your-dockerhub-username/ai-image-gen:gpu
```

### Local Development with Poetry

For local development, Poetry provides a convenient way to manage dependencies:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the application
python src/main.py
```

### RunPod Handler for Serverless Deployment

For serverless deployment, you may need to create a RunPod handler script that interfaces with the RunPod serverless infrastructure. Create a file named `runpod_handler.py` in the project root:

```python
import runpod
import subprocess
import time
import os

# Start the FastAPI server in the background
server_process = subprocess.Popen(
    ["python", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
)

# Wait for the server to start
time.sleep(5)

def handler(event):
    """
    RunPod serverless handler function.
    
    This function receives requests from the RunPod serverless infrastructure
    and forwards them to the FastAPI server.
    """
    try:
        # Extract request data
        method = event.get("method", "GET")
        path = event.get("path", "/")
        data = event.get("data", {})
        
        # Construct the URL
        url = f"http://localhost:8000{path}"
        
        # Make the request to the FastAPI server
        import requests
        response = requests.request(method, url, json=data)
        
        # Return the response
        return {
            "statusCode": response.status_code,
            "body": response.json()
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
```

The `runpod` package is already included in the `pyproject.toml` file. Make sure your Dockerfile is configured to use this handler for serverless deployments.

## Monitoring and Scaling

RunPod provides monitoring tools to track the usage and performance of your deployed endpoints. You can set up auto-scaling based on the load to ensure optimal performance and cost-efficiency.


## Usage

```json
{
    "input": {
        "method": "generate_image",
        "prompt": "A futuristic cityscape with flying cars and neon lights at night",
        "negative_prompt": "daylight, sun, bright, blurry, low quality",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "num_inference_steps": 30
    }
}
```

## Troubleshooting

If you encounter issues with the deployment, check the following:

1. Make sure your Docker image is built correctly and accessible
2. Verify that the GPU instance has enough VRAM for the models you're using
3. Check the logs in the RunPod console for any error messages
4. Ensure that your Hugging Face token is valid if you're using models that require authentication
