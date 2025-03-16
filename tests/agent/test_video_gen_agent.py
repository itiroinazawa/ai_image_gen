"""
Unit tests for the VideoGenerationAgent class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.agent.video_gen_agent import VideoGenerationAgent
from src.utils.config import Config


class TestVideoGenerationAgent(unittest.TestCase):
    """
    Test cases for the VideoGenerationAgent class.
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

        # Create a patch for the video generator
        self.video_generator_patch = patch("src.agent.video_gen_agent.VideoGenerator")
        self.mock_video_generator = self.video_generator_patch.start()

        # Create the agent
        self.agent = VideoGenerationAgent(self.config)

        # Mock the video generator instance
        self.mock_generator_instance = self.mock_video_generator.return_value

    def tearDown(self):
        """Tear down test fixtures."""
        self.video_generator_patch.stop()

        # Clean up test output directory
        for file in os.listdir(self.config.output_dir):
            os.remove(os.path.join(self.config.output_dir, file))
        os.rmdir(self.config.output_dir)

    def test_init(self):
        """Test the initialization of the agent."""
        self.assertEqual(self.agent.config, self.config)
        self.assertEqual(self.agent.video_generator, self.mock_generator_instance)
        self.assertTrue(os.path.exists(self.config.output_dir))

    def test_generate_video(self):
        """Test the generate_video method."""
        # Create a mock video object
        mock_video = MagicMock()
        self.mock_generator_instance.generate.return_value = mock_video

        # Mock uuid to get a predictable filename
        with patch("src.agent.video_gen_agent.uuid.uuid4", return_value="test-uuid"):
            # Call the method
            result = self.agent.generate_video(
                prompt="test prompt",
                negative_prompt="test negative prompt",
                model_id="test_model",
                num_inference_steps=20,
                guidance_scale=7.5,
                num_frames=16,
                height=512,
                width=512,
                seed=42,
            )

            # Verify the method was called with the correct arguments
            self.mock_generator_instance.generate.assert_called_once_with(
                prompt="test prompt",
                negative_prompt="test negative prompt",
                model_id="test_model",
                num_inference_steps=20,
                guidance_scale=7.5,
                num_frames=16,
                height=512,
                width=512,
                seed=42,
            )

            # Verify the result is a string (filename)
            self.assertEqual(result, "test-uuid.mp4")

            # Verify save_video was called
            self.mock_generator_instance.save_video.assert_called_once_with(
                mock_video, os.path.join(self.config.output_dir, "test-uuid.mp4")
            )

    def test_list_models(self):
        """Test the list_models method."""
        # Mock the return value of list_models
        mock_models = [
            {"id": "model1", "name": "Model 1", "type": "video"},
            {"id": "model2", "name": "Model 2", "type": "video"},
        ]
        self.mock_generator_instance.list_models.return_value = mock_models

        # Call the method
        result = self.agent.list_models()

        # Verify the method was called
        self.mock_generator_instance.list_models.assert_called_once()

        # Verify the result
        self.assertEqual(result, mock_models)


if __name__ == "__main__":
    unittest.main()
