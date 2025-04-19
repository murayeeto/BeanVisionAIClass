#!/bin/bash
# Setup and run the Bean Classification project

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download sample images
echo "Downloading sample images..."
python download_samples.py

# Run the web interface
echo "Starting the web interface..."
python app.py