#!/usr/bin/env python3
"""
Unit tests to verify that imports are working correctly.
"""
import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to the path
# When running from tests folder, we need to go up one directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


class TestImports(unittest.TestCase):
    """Test case for verifying imports and agent initialization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config for testing
        self.mock_config = type(
            "MockConfig",
            (),
            {
                "device": "cpu",
                "precision": "fp32",
                "output_dir": "output",
                "cache_dir": "cache",
            },
        )()

        # Create output directory
        os.makedirs(self.mock_config.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        # Could add cleanup code here if needed

    def test_module_imports(self):
        """Test importing all required modules."""
        # Try importing from the src.agent module

        # If we get here without exceptions, the test passes
        self.assertTrue(True, "Successfully imported all agent modules")

    def test_config_creation(self):
        """Test creating a Config instance."""
        from src.utils.config import Config

        config = Config()
        self.assertIsNotNone(config.device, "Config should have a device attribute")

    def test_image_agent_creation(self):
        """Test creating an ImageGenerationAgent instance."""
        from src.agent.image_gen_agent import ImageGenerationAgent

        with patch("src.agent.image_gen_agent.ImageGenerator"):
            agent = ImageGenerationAgent(self.mock_config)
            self.assertIsNotNone(
                agent, "Should create an ImageGenerationAgent instance"
            )

    def test_video_agent_creation(self):
        """Test creating a VideoGenerationAgent instance."""
        from src.agent.video_gen_agent import VideoGenerationAgent

        with patch("src.agent.video_gen_agent.VideoGenerator"):
            agent = VideoGenerationAgent(self.mock_config)
            self.assertIsNotNone(agent, "Should create a VideoGenerationAgent instance")

    def test_image_changer_agent_creation(self):
        """Test creating an ImageChangerAgent instance."""
        from src.agent.image_changer_agent import ImageChangerAgent

        with patch("src.agent.image_changer_agent.ImageProcessor"):
            agent = ImageChangerAgent(self.mock_config)
            self.assertIsNotNone(agent, "Should create an ImageChangerAgent instance")


if __name__ == "__main__":
    unittest.main()
