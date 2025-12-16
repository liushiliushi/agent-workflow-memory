#!/bin/bash

# Complete pipeline runner script
# Usage: ./run_pipeline.sh [website_name]
# Example: ./run_pipeline.sh shopping

# Get website parameter (default to "shopping" if not provided)
WEBSITE=${1:-"shopping"}

echo "Running WebArena pipeline for website: $WEBSITE"

# Set up environment variables
source ./setup_env.sh

echo "Starting pipeline with xvfb..."

# Run the pipeline with xvfb for headless browser support
xvfb-run -a python pipeline.py --website "$WEBSITE"

echo "Pipeline execution completed!"