from setuptools import find_packages, setup

# This setup.py is maintained for backward compatibility
# The project now uses Poetry for dependency management (see pyproject.toml)

setup(
    name="ai_image_gen",
    version="0.1.0",
    description="AI-powered agent for image processing and generation, as well as video generation from text prompts",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<3.14",
)
