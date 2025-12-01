#!/bin/bash
set -e

echo "Building C++ Extension..."
cmake -B build -S .
cmake --build build

echo "Starting Streamlit App..."
# Add build to PYTHONPATH so app.py can find bill_cuda.so
export PYTHONPATH=$PYTHONPATH:$(pwd)/build
streamlit run python/app.py
