#!/usr/bin/env python3
"""

Test runner for the AI Image & Video Generation Agent.
"""
import unittest
import sys
import os

# Add the project root to the path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all tests
from tests.agent.test_image_gen_agent import TestImageGenerationAgent
from tests.agent.test_video_gen_agent import TestVideoGenerationAgent
from tests.agent.test_image_changer_agent import TestImageChangerAgent
from tests.test_runpod_imports import TestRunpodImports
from tests.test_imports import TestImports


def run_all_tests():
    """Run all tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestImageGenerationAgent))
    test_suite.addTest(unittest.makeSuite(TestVideoGenerationAgent))
    test_suite.addTest(unittest.makeSuite(TestImageChangerAgent))
    test_suite.addTest(unittest.makeSuite(TestRunpodImports))
    test_suite.addTest(unittest.makeSuite(TestImports))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
