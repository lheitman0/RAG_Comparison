#!/bin/bash

# Check if Python virtual environment exists, create if it doesn't
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements if needed
if [ ! -f ".requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements_streamlit.txt
    touch .requirements_installed
fi

# Make sure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py