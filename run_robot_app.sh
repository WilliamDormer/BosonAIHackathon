#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "--------------------------------"
echo "🚀 Running the Main Application"
echo "--------------------------------"
echo "Check the Gradio interface at http://localhost:7861"
echo "--------------------------------"

# Set the environment variable for BOSON_API_KEY
USERNAME=$(head -n 1 src/config/secret_keys_vanessa.txt)
API_KEY=$(tail -n 1 src/config/secret_keys_vanessa.txt)

echo $USERNAME
echo $API_KEY

export BOSON_USERNAME=$USERNAME
export BOSON_API_KEY=$API_KEY
export OPENAI_API_KEY=$api_key

bash -c "python src/robot_app.py"