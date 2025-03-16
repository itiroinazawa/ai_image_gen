#!/bin/bash
# Docker build, tag, and push script for AI Image Generation RunPod Worker

# Configuration
DOCKER_REGISTRY="docker.io"  # Use your preferred registry (DockerHub, GitHub, etc.)
DOCKER_USERNAME="itiroinazawa"  # Replace with your Docker registry username
IMAGE_NAME="ai-image-gen-runpod-gpu"
VERSION=$(grep 'version' pyproject.toml | head -n 1 | cut -d '"' -f 2)  # Extract version from pyproject.toml
PLATFORM="linux/amd64"  # Default platform

# Parse command line arguments
NO_CACHE=false
USE_BUILDKIT=true
BUILD_ARGS="--build-arg DISABLE_MODEL_DOWNLOAD=1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-cache)
      NO_CACHE=true
      shift
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --build-arg)
      BUILD_ARGS="$BUILD_ARGS --build-arg $2"
      shift 2
      ;;
    --no-buildkit)
      USE_BUILDKIT=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Available options: --no-cache, --platform, --build-arg, --no-buildkit"
      exit 1
      ;;
  esac
done

# Full image name
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}"

# Tags
TAG_LATEST="${FULL_IMAGE_NAME}:latest"
TAG_VERSION="${FULL_IMAGE_NAME}:${VERSION}"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
TAG_TIMESTAMP="${FULL_IMAGE_NAME}:${VERSION}-${TIMESTAMP}"

# Set cache options
CACHE_OPTIONS=""
if [ "$NO_CACHE" = true ]; then
  CACHE_OPTIONS="--no-cache"
fi

# Enable BuildKit for better layer caching
if [ "$USE_BUILDKIT" = true ]; then
  export DOCKER_BUILDKIT=1
  BUILD_ARGS="$BUILD_ARGS --build-arg BUILDKIT_INLINE_CACHE=1"
fi

# Build the Docker image
echo "Building Docker image: ${TAG_VERSION}..."
echo "Using platform: ${PLATFORM}"
echo "Cache settings: ${CACHE_OPTIONS:-"Using default cache"}"
echo "BuildKit enabled: ${USE_BUILDKIT}"

docker build $CACHE_OPTIONS \
  --platform $PLATFORM \
  $BUILD_ARGS \
  -f Dockerfile.gpu \
  -t ${TAG_VERSION} \
  -t ${TAG_LATEST} \
  -t ${TAG_TIMESTAMP} \
  .

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
    docker push ${TAG_TIMESTAMP}
    
    echo "Docker images pushed successfully!"
    echo "Image tags:"
    echo "  - ${TAG_VERSION}"
    echo "  - ${TAG_LATEST}"
    echo "  - ${TAG_TIMESTAMP}"
else
    echo "Skipping push to Docker registry."
    echo "To push the image later, run:"
    echo "  docker login ${DOCKER_REGISTRY}"
    echo "  docker push ${TAG_VERSION}"
    echo "  docker push ${TAG_LATEST}"
    echo "  docker push ${TAG_TIMESTAMP}"
fi

# Print image size information
echo "\nImage size information:"
docker images ${FULL_IMAGE_NAME}
