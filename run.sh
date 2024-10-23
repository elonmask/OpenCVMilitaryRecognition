#!/bin/bash

# Create the virtual environment if it doesn't exist
if [ ! -d "military-detection" ]; then
    python3 -m venv --system-site-packages military-detection
fi

# Activate the virtual environment
source military-detection/bin/activate

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found, creating one with necessary dependencies."
    echo "opencv-python" > requirements.txt
    echo "ultralytics" >> requirements.txt
fi

# Install dependencies
pip install -r requirements.txt

# Run the main.py file located under /src/
python3 src/main.py