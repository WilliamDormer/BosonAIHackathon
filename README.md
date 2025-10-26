<h1 align="center">2025 BosonAI Hackathon</h1>
<p align="center">
  Making communication with AI as easy, natural and fun as talking to a human
</p>
<h2 align="center">The Quacks: AI-Enabled Assistive Devices for Disabled People</h2>

# Environment Configuration

## Prerequisites
- Python 3.11+ (recommended)
- Virtual environment (venv)

## Setup Instructions

### 1. Create and Activate Virtual Environment

if using uv: 
```bash
# Create virtual environment
python3 -m uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
```
if using conda/mamba
```bash
mamba create -n -y boson python=3.10
mamba activate boson
```

### 2. Install Dependencies
```bash
# Install from requirements.txt (minimal dependencies)
pip install -r requirements.txt
```
additional deps for robot arm: 
```bash
cd src/lerobot pip install -e . # install customized version of lerobot library.
cd cd ../.. # get back to top level.
```

### 3. Set Environment Variables
automatically:   
1. add file name ```secret_keys.txt``` under config/
2. first line is the username
3. second line is the api key  
Then the run script will source it automatically

or do it manually: 
```bash
# Set your Boson AI API key
export BOSON_API_KEY='your-api-key-here'
# Optional: Set custom runs directory
export VOICE_SIGHT_RUNS_DIR='/path/to/custom/runs'
```

### 4. Run the Application
for the Visual Assistant:
```bash
# Make sure you're in the project root and venv is activated
bash run_app.sh
```
for the Voice Robot Control
```bash
bash run_robot_app.sh
```

The application will start on `http://localhost:7861` by default.

# Api_doc
Check the api_doc for the detailed usage of our APIs.

# Available Models
- higgs-audio-generation-Hackathon
- Qwen3-32B-thinking-Hackathon
- Qwen3-32B-non-thinking-Hackathon
- Qwen3-14B-Hackathon
- higgs-audio-understanding-Hackathon
- Qwen3-Omni-30B-A3B-Thinking-Hackathon

# Supported voices for Audio Generation
- Supports voices: `belinda`, `broom_salesman`, `chadwick`, `en_man`, `en_woman`, `mabel`, `vex`, `zh_man_sichuan`


Google also provides free api keys thorugh https://aistudio.google.com/