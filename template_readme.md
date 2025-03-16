# AI Image & Video Generation RunPod Template

This RunPod template provides a serverless API for AI-powered image and video generation based on text prompts. It leverages state-of-the-art open-source models to create high-quality images and videos on demand.

## Features

- **Text-to-Image Generation**: Create stunning images from text descriptions using various diffusion models
- **Image Editing**: Modify existing images with text-guided transformations
- **Text-to-Video Generation**: Generate short videos from text prompts
- **Model Flexibility**: Support for multiple models and LoRA adapters
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Serverless Architecture**: Pay only for what you use with RunPod's serverless infrastructure

## Technical Specifications

- **Base Image**: NVIDIA CUDA 12.3.2 with cuDNN 9 on Ubuntu 22.04
- **Python Version**: 3.10
- **GPU Support**: Requires NVIDIA GPU with CUDA
- **Memory Requirement**: Minimum 16GB VRAM recommended
- **Storage**: At least 20GB for models and cache

## API Methods

### 1. Generate Image

Create an image from a text prompt.

**Parameters:**
- `prompt` (string, required): Text description of the desired image
- `negative_prompt` (string, optional): Text describing elements to avoid in the image
- `model_id` (string, optional): Model ID to use (default: `runwayml/stable-diffusion-v1-5`)
- `lora_id` (string, optional): LoRA adapter ID to use
- `num_inference_steps` (integer, optional): Number of denoising steps (default: 50)
- `guidance_scale` (float, optional): How closely to follow the prompt (default: 7.5)
- `height` (integer, optional): Image height in pixels (default: 512)
- `width` (integer, optional): Image width in pixels (default: 512)
- `seed` (integer, optional): Random seed for reproducibility

**Response:**
- URL to the generated image

### 2. Process Image

Modify an existing image using text prompts.

**Parameters:**
- `image` (string, required): Base64-encoded image or URL to image
- `prompt` (string, required): Text description of the desired modifications
- `negative_prompt` (string, optional): Text describing elements to avoid
- `model_id` (string, optional): Model ID to use
- `strength` (float, optional): Strength of the modification (default: 0.8)
- Other parameters similar to generate_image

**Response:**
- URL to the processed image

### 3. Generate Video

Create a short video from a text prompt.

**Parameters:**
- `prompt` (string, required): Text description of the desired video
- `negative_prompt` (string, optional): Text describing elements to avoid
- `model_id` (string, optional): Model ID to use (default: `damo-vilab/text-to-video-ms-1.7b`)
- `num_frames` (integer, optional): Number of frames to generate (default: 16)
- `height` (integer, optional): Video height in pixels (default: 256)
- `width` (integer, optional): Video width in pixels (default: 256)
- Other parameters similar to generate_image

**Response:**
- URL to the generated video

## Usage Examples

### Generate an Image

```json
{
  "input": {
    "method": "generate_image",
    "prompt": "A serene landscape with mountains and a lake at sunset",
    "negative_prompt": "people, buildings, text",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 512,
    "width": 768
  }
}
```

### Process an Existing Image

```json
{
  "input": {
    "method": "process_image",
    "image": "https://example.com/input-image.jpg",
    "prompt": "Convert to oil painting style",
    "strength": 0.7
  }
}
```

### Generate a Video

```json
{
  "input": {
    "method": "generate_video",
    "prompt": "A spaceship flying through a nebula",
    "num_frames": 24,
    "height": 256,
    "width": 256
  }
}
```

## Environment Variables

The template supports the following environment variables:

- `USE_GPU`: Set to 1 to enable GPU acceleration (default: 1)
- `USE_FP16`: Set to 1 to enable FP16 precision for faster inference (default: 1)
- `BUCKET_ENDPOINT_URL`: S3-compatible storage endpoint URL
- `BUCKET_ACCESS_KEY_ID`: Access key for S3 storage
- `BUCKET_SECRET_ACCESS_KEY`: Secret key for S3 storage
- `S3_BUCKET_NAME`: Fallback S3 bucket name

## Performance Considerations

- Image generation typically takes 5-15 seconds depending on the model and parameters
- Video generation can take 30-120 seconds depending on the number of frames and complexity
- Using fewer inference steps reduces quality but improves speed
- Lower resolution images and videos process faster

## Troubleshooting

### Common Issues

**Out of Memory Errors**
   - Reduce image/video dimensions
   - Use fewer inference steps
   - Try a smaller model

**Slow Generation**
   - Enable FP16 precision
   - Reduce the number of inference steps
   - Use a smaller model or lower resolution

**Storage Issues**
   - Ensure your S3 credentials are correctly configured
   - Check that the bucket exists and is accessible