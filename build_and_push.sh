#!/bin/bash
# Docker build, tag, and push script for AI Image Generation RunPod Worker

# Configuration
DOCKER_REGISTRY="docker.io"  # Use your preferred registry (DockerHub, GitHub, etc.)
DOCKER_USERNAME="itiroinazawa"  # Replace with your Docker registry username
IMAGE_NAME="ai-image-gen-runpod-gpu"
VERSION=$(grep 'version' pyproject.toml | head -n 1 | cut -d '"' -f 2)  # Extract version from pyproject.toml

# Full image name
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}"

# Tags
TAG_LATEST="${FULL_IMAGE_NAME}:latest"
TAG_VERSION="${FULL_IMAGE_NAME}:${VERSION}"

# Build the Docker image
echo "Building Docker image: ${TAG_VERSION}..."
docker build --build-arg DISABLE_MODEL_DOWNLOAD=1 --no-cache -f Dockerfile.gpu -t ${TAG_VERSION} -t ${TAG_LATEST} .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed!"
    exit 1
fi

echo "Docker image built successfully!"

# Ask if the user wants to push the image
read -p "Do you want to push the image to ${DOCKER_REGISTRY}? (y/n): " PUSH_CHOICE

if [ "$PUSH_CHOICE" = "y" ] || [ "$PUSH_CHOICE" = "Y" ]; then
    # Login to Docker registry
    echo "Logging in to Docker registry..."
    docker login ${DOCKER_REGISTRY}
    
    if [ $? -ne 0 ]; then
        echo "Error: Docker login failed!"
        exit 1
    fi
    
    # Push the images
    echo "Pushing Docker image with all tags..."
    docker push ${TAG_VERSION}
    docker push ${TAG_LATEST}
    
    echo "Docker images pushed successfully!"
    echo "Image tags:"
    echo "  - ${TAG_VERSION}"
    echo "  - ${TAG_LATEST}"
else
    echo "Skipping push to Docker registry."
    echo "To push the image later, run:"
    echo "  docker login ${DOCKER_REGISTRY}"
    echo "  docker push ${TAG_VERSION}"
    echo "  docker push ${TAG_LATEST}"
fi
