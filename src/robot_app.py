"""
Voice Robot Controller.
"""

import os
import sys
import glob
from typing import Optional
from datetime import datetime

try:
    import gradio as gr  # type: ignore
except ImportError:
    print("Error: gradio is not installed. Please install it with: pip install gradio")
    sys.exit(1)

from utils.logger import RunLogger, create_logger  # type: ignore
from utils.voice_robot.agent import VoiceRobotAgent

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class VoiceRobotApp:
    
    def __init__(self, api_key: Optional[str] = None, runs_dir: str = "runs"):
        self.api_key = api_key or os.getenv("BOSON_API_KEYs")
        self.runs_dir = runs_dir

        # Ensure runs directory exists
        os.makedirs(runs_dir, exist_ok=True)
        
        # Current logger and session
        run_name = f"voice_robot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        self.current_logger = create_logger(run_name=run_name, base_dir=self.runs_dir)
        
        # Get available prompt systems
        self.available_prompts = self._get_available_prompts()

        # Initialize agent (will be created with logger in start_session)
        self.agent: VoiceRobotAgent = VoiceRobotAgent(
            prompt_system=self.available_prompts[0][1] if self.available_prompts else "prompt.yaml",
            api_key=self.api_key,
            logger=self.current_logger,
        )
    
    def _get_available_prompts(self) -> list:
        """Get list of available prompt systems from the prompts directory."""
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts", "robot")
        yaml_files = glob.glob(os.path.join(prompts_dir, "*.yaml"))
        
        prompt_options = []
        for yaml_file in yaml_files:
            filename = os.path.basename(yaml_file)
            name = filename.replace(".yaml", "").replace("_", " ").title()
            prompt_options.append((name, filename))
        
        return prompt_options

    def build_interface(self):
        """Build the Gradio interface."""

        with gr.Blocks(title="Robot Voice Controller", theme=gr.themes.Soft()) as demo:
            # Audio input
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Tell the robot what to do:")
                # Audio input
                audio_input = gr.Audio(
                    label="Record Your Message",
                    type="filepath",
                    sources=["microphone"],
                    format="wav"
                )

                # Prompt system selection

                prompt_dropdown = gr.Dropdown(
                    choices=self.available_prompts,
                    value=self.available_prompts[0][1] if self.available_prompts else "Please Select a prompt",
                    label="Prompt System",
                    info="Select the AI personality and capabilities"
                )

                # Process button
                with gr.Row():
                    answer_btn = gr.Button("🎯 Execute", variant="primary")

            with gr.Column():
                gr.Markdown("### 🔧 Execution Log")
                execution_log = gr.Textbox(
                    label="System Log",
                    interactive=False,
                    lines=6,
                    value="System ready."
                )
            
            # Store components for event handlers.
            self.components = {
                "audio_input" : audio_input,
                "execution_log" : execution_log,
                "prompt_dropdown" : prompt_dropdown,
            }
            
            answer_btn.click(
                fn=self.process_input,
                inputs=[audio_input,prompt_dropdown],
                outputs=[execution_log]
            ) 
        return demo
    
    def process_input(self, audio_input: str, prompt_system: str):
        if not self.agent:
            raise Exception("self.Agent was none.")
        
        # override the system prompt for the agent according to selected value.
        self.agent.prompt_system = prompt_system

        # prompt system is a string name of the file path. 
        # strip the suffix to get the mode ( the name must be the  mode name.)
        mode = prompt_system.removesuffix(".yaml")
        if not audio_input:
            return None, "Please provide an audio input."

        # ask the agent to do something with the audio input
        self.agent.process_input(audio_input, mode=mode)
    
    def launch(self, **kwargs):
        """Launch the gradio interface"""
        demo = self.build_interface()
        kwargs.setdefault("show_api", False)
        demo.launch(**kwargs)
    
def main():
    """Main entry point for the application"""

    api_key = os.getenv("BOSON_API_KEY")

    if not api_key:
        raise Exception("You MUST supply a BOSON_API_KEY in the environment.")

    # setup runs directory: 
    project_root = os.path.dirname(current_dir)
    runs_dir = os.path.join(project_root, "runs", "voice_robot")

    app = VoiceRobotApp(api_key=api_key, runs_dir = runs_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )

if __name__ == "__main__":
    main()
            




            
            



    
