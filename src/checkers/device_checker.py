import torch


def verify_device(logger, requested_device: str) -> torch.device:
    """
    Verify if the requested device is available and return an appropriate device.

    Args:
        requested_device: The device requested in the config

    Returns:
        Available torch device
    """
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if (
        requested_device == "mps"
        and not hasattr(torch.backends, "mps")
        and not getattr(torch.backends, "mps", {}).get("is_available", False)
    ):
        logger.warning(
            "MPS (Apple Silicon) requested but not available. Falling back to CPU."
        )
        return torch.device("cpu")

    try:
        device = torch.device(requested_device)
        # Test the device by creating a small tensor
        torch.zeros(1, device=device)
        logger.info(f"Successfully initialized device: {device}")
        return device
    except Exception as e:
        logger.warning(
            f"Error initializing device {requested_device}: {str(e)}. Falling back to CPU."
        )
        return torch.device("cpu")


def verify_precision(
    logger, requested_device: torch.device, requested_precision: str
) -> str:
    """
    Verify if the requested precision is supported on the current device.

    Args:
        requested_precision: The precision requested in the config

    Returns:
        Supported precision string
    """
    # If using CPU, always use fp32
    if requested_device.type == "cpu" and requested_precision == "fp16":
        logger.warning("FP16 precision requested on CPU. Falling back to FP32.")
        return "fp32"

    # If using CUDA, check if FP16 is supported
    if requested_device.type == "cuda" and requested_precision == "fp16":
        try:
            # Test if autocast works
            with torch.autocast("cuda", enabled=True):
                torch.zeros(1, device=requested_device)
            return "fp16"
        except Exception as e:
            logger.warning(
                f"FP16 precision not supported on this CUDA device: {str(e)}. Falling back to FP32."
            )
            return "fp32"

    return requested_precision
