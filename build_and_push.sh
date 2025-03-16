#!/bin/bash
# Optimized Docker build, tag, and push script for AI Image Generation RunPod Worker

set -e  # Exit script on error

# Configuration
DOCKER_REGISTRY="docker.io"
DOCKER_USERNAME="itiroinazawa"
IMAGE_NAME="ai-image-gen-runpod-gpu"
VERSION=$(grep 'version' pyproject.toml | head -n 1 | cut -d '"' -f 2)
PLATFORM="linux/amd64"
USE_BUILDKIT=true
NO_CACHE=false
BUILD_ARGS="--build-arg DISABLE_MODEL_DOWNLOAD=1"

# Parse command-line arguments using getopts
while getopts ":np:a:b" opt; do
  case ${opt} in
    n ) NO_CACHE=true ;;
    p ) PLATFORM="$OPTARG" ;;
    a ) BUILD_ARGS+=" --build-arg $OPTARG" ;;
    b ) USE_BUILDKIT=false ;;
    \? ) echo "Usage: $0 [-n] [-p platform] [-a build_arg] [-b]" && exit 1 ;;
  esac
done

# Full image name and tags
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}"
TAG_LATEST="${FULL_IMAGE_NAME}:latest"
TAG_VERSION="${FULL_IMAGE_NAME}:${VERSION}"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
TAG_TIMESTAMP="${FULL_IMAGE_NAME}:${VERSION}-${TIMESTAMP}"
TAGS=("$TAG_VERSION" "$TAG_LATEST" "$TAG_TIMESTAMP")

# Set cache options
CACHE_OPTIONS=$([ "$NO_CACHE" = true ] && echo "--no-cache" || echo "")

# Enable BuildKit for faster builds
if [ "$USE_BUILDKIT" = true ]; then
  export DOCKER_BUILDKIT=1
  BUILD_ARGS+=" --build-arg BUILDKIT_INLINE_CACHE=1"
fi

echo "🚀 Starting Docker build..."
echo "🔹 Platform: $PLATFORM"
echo "🔹 Cache: $([ "$NO_CACHE" = true ] && echo "Disabled" || echo "Enabled")"
echo "🔹 BuildKit: $([ "$USE_BUILDKIT" = true ] && echo "Enabled" || echo "Disabled")"
echo "🔹 Tags: ${TAGS[*]}"

# Build the Docker image with multiple tags
docker build $CACHE_OPTIONS --platform $PLATFORM $BUILD_ARGS -f Dockerfile.gpu \
  $(printf " -t %s" "${TAGS[@]}") .

echo "✅ Docker build completed successfully!"

# Ask the user if they want to push
read -p "Do you want to push the image? (y/n): " PUSH_CHOICE
if [[ "$PUSH_CHOICE" =~ ^[Yy]$ ]]; then
  echo "🔐 Logging into Docker registry..."
  docker login $DOCKER_REGISTRY

  echo "📤 Pushing Docker images..."
  for TAG in "${TAGS[@]}"; do
    docker push "$TAG" &
  done
  wait  # Ensure all parallel pushes complete

  echo "✅ Docker images pushed successfully!"
else
  echo "❌ Skipping push. To push later, run:"
  for TAG in "${TAGS[@]}"; do
    echo "  docker push $TAG"
  done
fi

# Print image details
echo -e "\n📊 Image size information:"
docker images "$FULL_IMAGE_NAME"