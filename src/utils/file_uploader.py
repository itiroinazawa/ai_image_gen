import os

import boto3
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils import upload_file_to_bucket


# =============================================================================
# Utility Functions for File Saving
# =============================================================================


def save_file(logger: RunPodLogger, output_path):
    
    try:
        # If an output file was generated, attempt to upload it and attach a public URL
        if output_path:
            logger.info(f"Output path: {output_path}")
            presigned_url = _save_file_runpod(logger, output_path)

            if presigned_url:
                logger.info(f"RunPod file saved successfully: {output_path}")
                return presigned_url
            else:
                presigned_url = _save_file_s3(logger, output_path)
                logger.info(f"S3 file saved successfully: {output_path}")
                return presigned_url
        else:
            logger.warn("No output path provided, skipping file upload.")
            return None
    except:
        presigned_url = _save_file_s3(logger, output_path)
        logger.info(f"Fallback - S3 file saved successfully: {output_path}")
        return presigned_url


def _save_file_runpod(logger: RunPodLogger, output_path):
    bucket_url = os.environ.get("BUCKET_ENDPOINT_URL")
    bucket_access_key = os.environ.get("BUCKET_ACCESS_KEY_ID")
    bucket_secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")

    bucket_creds = {
        "endpointUrl": bucket_url,
        "accessId": bucket_access_key,
        "accessSecret": bucket_secret_key,
    }

    logger.info("Saving File...")

    if output_path:
        if os.path.exists(output_path):
            logger.info(f"File {output_path} exists, uploading to storage...")
            filename = os.path.basename(output_path)
            presigned_url = upload_file_to_bucket(filename, output_path, bucket_creds)
            return presigned_url
        else:
            logger.warn(f"File {output_path} does not exist, skipping upload.")
            return None
    else:
        logger.warn("No output path provided, skipping file upload.")
        return None


def _save_file_s3(logger: RunPodLogger, output_path):
    filename = os.path.basename(output_path)
    bucket_name = os.getenv("S3_BUCKET_NAME")

    s3_client = boto3.client("s3")
    s3_client.upload_file(output_path, bucket_name, filename)

    logger.info(f"File {filename} uploaded to S3 bucket: {bucket_name}")
    return f"https://{bucket_name}.s3.amazonaws.com/{filename}"
