autoflake --remove-all-unused-imports --recursive --in-place .
isort .
black .
vulture . > unused_code.txt
find . -type d -name "__pycache__" -exec rm -r {} +