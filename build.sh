#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Generate the machine learning models (since they are ignored from git)
echo "Generating First Innings Score Model..."
python generate_model.py

echo "Generating Win Probability Model..."
python generate_win_model.py

echo "Generating pre-processed datasets to speed up loading..."
python generate_cache.py

# Clear pip build cache to free disk space after build
pip cache purge || true

echo "Build complete!"
