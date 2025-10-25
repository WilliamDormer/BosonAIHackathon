#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "--------------------------------"
echo "🚀 Running the Main Application"
echo "--------------------------------"
echo "Check the Gradio interface at http://localhost:7860"
echo "--------------------------------"

api_key="bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
export BOSON_API_KEY=$api_key
export OPENAI_API_KEY=$api_key
bash -c "python src/app.py"