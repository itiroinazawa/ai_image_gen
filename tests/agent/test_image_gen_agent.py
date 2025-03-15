"""
Unit tests for the ImageGenerationAgent class.
"""
import os
import unittest
from unittest.mock import patch, MagicMock, ANY
from PIL import Image
import random

from src.agent.image_gen_agent import ImageGenerationAgent
from src.utils.config import Config


class TestImageGenerationAgent(unittest.TestCase):
    """
    Test cases for the ImageGenerationAgent class.
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
        
        # Create a patch for the image generator
        self.image_generator_patch = patch('src.agent.image_gen_agent.ImageGenerator')
        self.mock_image_generator = self.image_generator_patch.start()
        
        # Create the agent
        self.agent = ImageGenerationAgent(self.config)
        
        # Mock the image generator instance
        self.mock_generator_instance = self.mock_image_generator.return_value

    def tearDown(self):
        """Tear down test fixtures."""
        self.image_generator_patch.stop()
        
        # Clean up test output directory
        for file in os.listdir(self.config.output_dir):
            os.remove(os.path.join(self.config.output_dir, file))
        os.rmdir(self.config.output_dir)

    def test_init(self):
        """Test the initialization of the agent."""
        self.assertEqual(self.agent.config, self.config)
        self.assertEqual(self.agent.image_generator, self.mock_generator_instance)
        self.assertTrue(os.path.exists(self.config.output_dir))

    def test_generate_image(self):
        """Test the generate_image method."""
        # Create a mock image
        mock_image = MagicMock(spec=Image.Image)
        self.mock_generator_instance.generate.return_value = mock_image
        
        # Call the method
        result = self.agent.generate_image(
            prompt="test prompt",
            negative_prompt="test negative prompt",
            model_id="test_model",
            lora_id="test_lora",
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42
        )
        
        # Verify the method was called with the correct arguments
        self.mock_generator_instance.generate.assert_called_once_with(
            prompt="test prompt",
            negative_prompt="test negative prompt",
            model_id="test_model",
            lora_id="test_lora",
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42
        )
        
        # Verify the result is a string (filename)
        self.assertIsInstance(result, str)

    def test_list_models(self):
        """Test the list_models method."""
        # Mock the return value of list_models
        mock_models = [
            {"id": "model1", "name": "Model 1", "type": "image"},
            {"id": "model2", "name": "Model 2", "type": "image"}
        ]
        self.mock_generator_instance.list_models.return_value = mock_models
        
        # Call the method
        result = self.agent.list_models()
        
        # Verify the method was called
        self.mock_generator_instance.list_models.assert_called_once()
        
        # Verify the result
        self.assertEqual(result, mock_models)

    def test_list_loras(self):
        """Test the list_loras method."""
        # Mock the return value of list_loras
        mock_loras = [
            {"id": "lora1", "name": "LoRA 1"},
            {"id": "lora2", "name": "LoRA 2"}
        ]
        self.mock_generator_instance.list_loras.return_value = mock_loras
        
        # Call the method
        result = self.agent.list_loras()
        
        # Verify the method was called
        self.mock_generator_instance.list_loras.assert_called_once()
        
        # Verify the result
        self.assertEqual(result, mock_loras)

    @patch('src.agent.image_gen_agent.uuid.uuid4')
    def test_save_image(self, mock_uuid):
        """Test the _save_image method."""
        # Mock UUID
        mock_uuid.return_value = "test-uuid"
        
        # Create a mock image
        mock_image = MagicMock(spec=Image.Image)
        
        # Call the method (it's private, but we can still test it)
        result = self.agent._save_image(mock_image)
        
        # Verify the image was saved
        mock_image.save.assert_called_once_with(os.path.join(self.config.output_dir, "test-uuid.png"))
        
        # Verify the result is the expected filename
        self.assertEqual(result, "test-uuid.png")


if __name__ == '__main__':
    unittest.main()
