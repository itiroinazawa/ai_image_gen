"""
Unit tests for the ImageChangerAgent class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import UploadFile
from PIL import Image

from src.agent.image_changer_agent import ImageChangerAgent
from src.utils.config import Config


class TestImageChangerAgent(unittest.TestCase):
    """
    Test cases for the ImageChangerAgent class.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = MagicMock(spec=Config)
        self.config.device = "cpu"
        self.config.precision = "fp32"
        self.config.output_dir = "test_output"
        self.config.cache_dir = "test_cache"

        # Ensure the test output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Create a patch for the image processor
        self.image_processor_patch = patch(
            "src.agent.image_changer_agent.ImageProcessor"
        )
        self.mock_image_processor = self.image_processor_patch.start()

        # Create the agent
        self.agent = ImageChangerAgent(self.config)

        # Mock the image processor instance
        self.mock_processor_instance = self.mock_image_processor.return_value

    def tearDown(self):
        """Tear down test fixtures."""
        self.image_processor_patch.stop()

        # Clean up test output directory
        for file in os.listdir(self.config.output_dir):
            os.remove(os.path.join(self.config.output_dir, file))
        os.rmdir(self.config.output_dir)

    def test_init(self):
        """Test the initialization of the agent."""
        self.assertEqual(self.agent.config, self.config)
        self.assertEqual(self.agent.image_processor, self.mock_processor_instance)
        self.assertTrue(os.path.exists(self.config.output_dir))

    @patch("src.agent.image_changer_agent.random.randint")
    def test_process_image(self, mock_randint):
        """Test the process_image method."""
        # Mock random seed
        mock_randint.return_value = 42

        # Create a mock image
        mock_input_image = MagicMock(spec=Image.Image)
        mock_processed_image = MagicMock(spec=Image.Image)

        # Mock the image processor
        self.mock_processor_instance.process.return_value = mock_processed_image

        # Create a mock UploadFile
        mock_file = MagicMock()
        mock_file.read.return_value = b"test_image_data"
        mock_upload_file = MagicMock(spec=UploadFile)
        mock_upload_file.file = mock_file

        # Mock PIL Image.open
        with patch(
            "src.agent.image_changer_agent.Image.open", return_value=mock_input_image
        ):
            # Call the method
            result = self.agent.process_image(
                image=mock_upload_file,
                prompt="test prompt",
                negative_prompt="test negative prompt",
                model_id="test_model",
                lora_id="test_lora",
                num_inference_steps=50,
                guidance_scale=7.5,
                strength=0.8,
                seed=None,
            )

        # Verify the image processor was called with the correct arguments
        self.mock_processor_instance.process.assert_called_once_with(
            image=mock_input_image,
            prompt="test prompt",
            negative_prompt="test negative prompt",
            model_id="test_model",
            lora_id="test_lora",
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.8,
            seed=42,
        )

        # Verify the result is a string (filename)
        self.assertIsInstance(result, str)

    @patch("src.agent.image_changer_agent.random.randint")
    @patch("src.agent.image_changer_agent.requests.get")
    def test_process_image_url(self, mock_get, mock_randint):
        """Test the process_image_url method."""
        # Mock random seed
        mock_randint.return_value = 42

        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = b"test_image_data"
        mock_get.return_value = mock_response

        # Create mock images
        mock_input_image = MagicMock(spec=Image.Image)
        mock_processed_image = MagicMock(spec=Image.Image)

        # Mock the image processor
        self.mock_processor_instance.process.return_value = mock_processed_image

        # Mock PIL Image.open
        with patch(
            "src.agent.image_changer_agent.Image.open", return_value=mock_input_image
        ):
            # Call the method
            result = self.agent.process_image_url(
                image_url="https://example.com/test.jpg",
                prompt="test prompt",
                negative_prompt="test negative prompt",
                model_id="test_model",
                lora_id="test_lora",
                num_inference_steps=50,
                guidance_scale=7.5,
                strength=0.8,
                seed=None,
            )

        # Verify requests.get was called with the correct URL
        mock_get.assert_called_once_with("https://example.com/test.jpg")

        # Verify the image processor was called with the correct arguments
        self.mock_processor_instance.process.assert_called_once_with(
            image=mock_input_image,
            prompt="test prompt",
            negative_prompt="test negative prompt",
            model_id="test_model",
            lora_id="test_lora",
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.8,
            seed=42,
        )

        # Verify the result is a string (filename)
        self.assertIsInstance(result, str)

    @patch("src.agent.image_changer_agent.uuid.uuid4")
    def test_save_image(self, mock_uuid):
        """Test the _save_image method."""
        # Mock UUID
        mock_uuid.return_value = "test-uuid"

        # Create a mock image
        mock_image = MagicMock(spec=Image.Image)

        # Call the method (it's private, but we can still test it)
        result = self.agent._save_image(mock_image)

        # Verify the image was saved
        mock_image.save.assert_called_once_with(
            os.path.join(self.config.output_dir, "test-uuid.png")
        )

        # Verify the result is the expected filename
        self.assertEqual(result, "test-uuid.png")


if __name__ == "__main__":
    unittest.main()
