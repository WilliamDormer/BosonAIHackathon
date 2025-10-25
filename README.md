<h1 align="center">2025 BosonAI Hackathon</h1>

TODO:

- [x] Build general-purpose agent pipeline
- [x] Integrate comprehensive logging
- [ ] Fix Gradio UI
- [ ] Refine prompts and tool-calling logics
- [ ] Make audio interaction more natural and full of emotion
- [ ] Implement audio-driven robot arm manipulation demo
- [ ] Code refactoring and README writing

<p align="center">
  Making communication with AI as easy, natural and fun as talking to a human
</p>

# Environment Configuration

## Prerequisites
- Python 3.11+ (recommended)
- Virtual environment (venv)

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install from requirements.txt (minimal dependencies)
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# Set your Boson AI API key
export BOSON_API_KEY='your-api-key-here'

# Optional: Set custom runs directory
export VOICE_SIGHT_RUNS_DIR='/path/to/custom/runs'
```

### 4. Run the Application
```bash
# Make sure you're in the project root and venv is activated
bash run_app.sh
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