#!/bin/bash

# Check if Python 3.10.12 is installed
PYTHON_VERSION=$(python3 --version 2>&1)
REQUIRED_VERSION="Python 3.10.12"

if [[ $PYTHON_VERSION != $REQUIRED_VERSION ]]; then
    echo "Python 3.10.12 is not installed. Please install it first."
    exit 1
fi

# Define the virtual environment directory
VENV_DIR="venv"

# Create a virtual environment using Python 3.10.12
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    # Install the required packages
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
    deactivate
    exit 1
fi

# Deactivate the virtual environment
deactivate

echo "Virtual environment setup complete. To activate it, run 'source $VENV_DIR/bin/activate'"
