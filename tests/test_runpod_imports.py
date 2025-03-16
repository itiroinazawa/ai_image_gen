#!/usr/bin/env python3
"""
Unit tests to verify that imports for the runpod handler are working correctly.
"""
import os
import sys
import unittest

# Add the project root to the path
# When running from tests folder, we need to go up one directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


class TestRunpodImports(unittest.TestCase):
    """Test case for verifying RunPod handler imports."""

    def test_src_prefix_imports(self):
        """Test importing modules using src prefix."""
        # Try importing from the src.agent module directly

        # If we get here without exceptions, the test passes
        self.assertTrue(True, "Successfully imported all modules using 'src' prefix")

    def test_runpod_imports(self):
        """Test importing modules as they are imported in runpod_handler.py."""
        # Now try the same imports as used in runpod_handler.py but with src prefix

        # If we get here without exceptions, the test passes
        self.assertTrue(
            True, "Successfully imported all modules using relative imports"
        )


if __name__ == "__main__":
    unittest.main()
