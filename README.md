# AI Image & Video Generation Agent

An AI-powered agent for image processing and generation, as well as video generation from text prompts.

## Features

- **Image Processing**
  - Upload images or provide image links
  - Apply modifications using checkpoint models, LoRAs, and text-based prompts
  
- **Prompt-to-Image Generation**
  - Generate images from text prompts using open-source models
  
- **Prompt-to-Video Generation**
  - Generate short videos from text prompts

## Tech Stack

- Python (primary language)
- Langchain (for workflow orchestration)
- Open-source models for image and video generation
- Optional: FLUX [dev] for additional capabilities

## Setup

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Docker (for containerization)
- RunPod account (for deployment)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai_image_gen
   ```

2. Install dependencies using Poetry:
   ```
   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   ```

3. Configure environment variables:
   ```
   cp .env.example .env
   # Edit .env file with your configuration
   ```

## Usage

### Local Development

```bash
# Activate the Poetry virtual environment
poetry shell

# Run the application
python src/main.py
```

### Docker Deployment

```bash
docker build -t ai-image-gen .
docker run -p 8000:8000 ai-image-gen
```

### RunPod Deployment

Follow the instructions in the `deployment/README.md` file.

## Project Structure

```
ai_image_gen/
├── src/                    # Source code
│   ├── agent/              # Agent implementation
│   ├── models/             # Model interfaces
│   ├── processors/         # Image and video processors
│   ├── utils/              # Utility functions
│   └── main.py             # Entry point
├── tests/                  # Test suite
├── deployment/             # Deployment configurations
│   ├── Dockerfile          # Main Dockerfile
│   ├── Dockerfile.gpu      # GPU-optimized Dockerfile
│   └── README.md           # Deployment instructions
├── examples/               # Example usage
├── .env.example            # Example environment variables
├── pyproject.toml          # Poetry configuration and dependencies
└── README.md               # This file
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
