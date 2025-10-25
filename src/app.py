"""
Voice Sight - Audio-based Agentic Pipeline with Gradio Interface.
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

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from utils.voice_sight.agent import VoiceSightAgent  # type: ignore  # noqa: E402
from utils.voice_sight.session import VoiceSightSession  # type: ignore  # noqa: E402
from utils.logger import RunLogger, create_logger  # type: ignore  # noqa: E402


class VoiceSightApp:
    """Main application class for Voice Sight audio Agentic pipeline."""
    
    def __init__(self, api_key: Optional[str] = None, runs_dir: str = "runs"):
        """Initialize the application."""
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.runs_dir = runs_dir
        
        # Ensure runs directory exists
        os.makedirs(runs_dir, exist_ok=True)
        
        # Current logger and session
        self.current_logger: Optional[RunLogger] = None
        self.current_session: Optional[VoiceSightSession] = None
        self.current_prompt_system: Optional[str] = None
        
        # Initialize agent (will be created with logger in start_session)
        self.agent: Optional[VoiceSightAgent] = None
        
        # Session viewer state
        self.loaded_session_data: Optional[dict] = None
        self.current_interaction_index: int = 0
        
        # Get available prompt systems
        self.available_prompts = self._get_available_prompts()
    
    def _get_available_prompts(self) -> list:
        """Get list of available prompt systems from the prompts directory."""
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        yaml_files = glob.glob(os.path.join(prompts_dir, "*.yaml"))
        
        prompt_options = []
        for yaml_file in yaml_files:
            filename = os.path.basename(yaml_file)
            name = filename.replace(".yaml", "").replace("_", " ").title()
            prompt_options.append((name, filename))
        
        return prompt_options
    
    def build_interface(self):
        """Build the Gradio interface."""
        
        with gr.Blocks(title="Voice Sight - Audio Agentic Pipeline", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎙️ Voice Sight - Audio Agentic Pipeline")
            gr.Markdown("An intelligent audio interface that understands and responds through speech.")
            
            with gr.Tab("Interactive Agent"):
                gr.Markdown("### 🎤 Audio Input")
                with gr.Row():
                    with gr.Column(scale=1):
                        
                        # Prompt system selection
                        prompt_dropdown = gr.Dropdown(
                            choices=self.available_prompts,
                            value=self.available_prompts[0][1] if self.available_prompts else "voice_sight.yaml",
                            label="Prompt System",
                            info="Select the AI personality and capabilities"
                        )
                        
                        # Audio input
                        audio_input = gr.Audio(
                            label="Record Your Message",
                            type="filepath",
                            sources=["microphone"],
                            format="wav",
                            container=True
                        )
                        
                        # Image input for visual analysis
                        image_input = gr.Image(
                            label="Upload Image for Visual Analysis",
                            type="filepath",
                            sources=["upload", "webcam"],
                            height=380,
                        )
                        
                        # Process button (first click will auto-start session)
                        with gr.Row():
                            answer_btn = gr.Button("🎯 Answer", variant="primary")
                            reset_btn = gr.Button("🔄 Reset", variant="secondary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔊 Audio Response")
                        gr.Markdown("*Play the agent's audio response*")
                        
                        # Audio output
                        audio_output = gr.Audio(
                            label="Agent Audio Response",
                            type="filepath",
                            interactive=False
                        )
                            
                        gr.Markdown("### 🔧 Execution Log")
                        execution_log = gr.Textbox(
                            label="System Log",
                            interactive=False,
                            lines=6,
                            value="System ready."
                        )
                        
            with gr.Tab("Runs & Sessions"):
                gr.Markdown("### 📂 Manage Sessions")
                gr.Markdown("Load and view previous sessions from the runs directory.")
            
                # Sessions loader bar
                with gr.Row():
                    session_dropdown = gr.Dropdown(
                        choices=self._list_sessions(),
                        value=None,
                        label="Select a session to view",
                        info="Sessions are saved under the runs/ directory"
                    )
                    with gr.Column():
                        refresh_sessions_btn = gr.Button("🔄 Refresh", variant="primary")
                        load_session_btn = gr.Button("📂 Load Session", variant="primary")
                
                # Session viewer - step-wise interaction display
                gr.Markdown("---")
                gr.Markdown("### 🎭 Session Replay")
                
                with gr.Row():
                    interaction_counter = gr.Markdown("No session loaded", elem_id="interaction_counter")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🗣️ User Input")
                        user_transcript_display = gr.Textbox(
                            label="Audio Transcript",
                            interactive=False,
                            lines=3,
                            placeholder="User's spoken input will appear here"
                        )
                        user_image_display = gr.Image(
                            label="User Image (if any)",
                            interactive=False,
                            height=200
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🤖 Model Response")
                        model_response_display = gr.Textbox(
                            label="Response Text",
                            interactive=False,
                            lines=3,
                            placeholder="Model's response will appear here"
                        )
                        model_audio_display = gr.Audio(
                            label="Generated Audio Response",
                            interactive=False,
                            type="filepath"
                        )
                
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", size="sm")
                    next_btn = gr.Button("➡️ Next", size="sm")

            # Store components for event handlers
            self.components = {
                'audio_input': audio_input,
                'image_input': image_input,
                'audio_output': audio_output,
                # 'status_display': status_display,
                'execution_log': execution_log,
                'prompt_dropdown': prompt_dropdown,
                'session_dropdown': session_dropdown,
                'interaction_counter': interaction_counter,
                'user_transcript_display': user_transcript_display,
                'user_image_display': user_image_display,
                'model_response_display': model_response_display,
                'model_audio_display': model_audio_display
            }
            
            # Event handlers - Interactive Agent tab
            reset_btn.click(
                fn=self.reset_session,
                inputs=[],
                outputs=[audio_input, image_input, audio_output, execution_log]
            )
            
            answer_btn.click(
                fn=self.process_input,
                inputs=[audio_input, image_input, prompt_dropdown],
                outputs=[audio_output, execution_log]
            )

            # Prompt system change should reset state and clear UI
            prompt_dropdown.change(
                fn=self.on_prompt_change,
                inputs=[prompt_dropdown],
                outputs=[audio_input, image_input, audio_output, execution_log]
            )

            # Sessions tab event handlers
            refresh_sessions_btn.click(
                fn=self.refresh_sessions,
                inputs=[],
                outputs=[session_dropdown]
            )
            
            load_session_btn.click(
                fn=self.load_session_and_display,
                inputs=[session_dropdown],
                outputs=[
                    interaction_counter,
                    user_transcript_display,
                    user_image_display,
                    model_response_display,
                    model_audio_display
                ]
            )
            
            prev_btn.click(
                fn=self.navigate_previous,
                inputs=[],
                outputs=[
                    interaction_counter,
                    user_transcript_display,
                    user_image_display,
                    model_response_display,
                    model_audio_display
                ]
            )
            
            next_btn.click(
                fn=self.navigate_next,
                inputs=[],
                outputs=[
                    interaction_counter,
                    user_transcript_display,
                    user_image_display,
                    model_response_display,
                    model_audio_display
                ]
            )
        
        return demo
    
    def _init_session(self, prompt_system: str, with_greeting: bool = False):
        """Initialize a new session and agent, optionally generating a greeting."""
        # Finalize old logger if any
        if self.current_logger:
            try:
                self.current_logger.finalize()
            except Exception:
                pass
        # Create new logger and session
        run_name = f"voice_sight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        self.current_logger = create_logger(run_name=run_name, base_dir=self.runs_dir)
        self.current_session = VoiceSightSession()
        # Create agent with logger and selected prompt system
        self.agent = VoiceSightAgent(
            api_key=self.api_key,
            use_thinking=True,
            logger=self.current_logger,
            prompt_system=prompt_system
        )
        self.agent.clear_conversation_context()
        # Track current prompt system
        self.current_prompt_system = prompt_system
        # Log session start
        self.current_logger.log_step("session_start", {
            "session_id": self.current_session.session_id,
            "timestamp": datetime.now().isoformat(),
            "prompt_system": prompt_system
        })
        # Optional greeting
        if with_greeting:
            greeting_result = self.agent.start_conversation()
            if greeting_result.get("success"):
                self.current_session.add_message(
                    role="assistant",
                    content=greeting_result.get("text", ""),
                    metadata={"audio_path": greeting_result.get("audio_path")}
                )
        return True
    
    def _save_current_session(self):
        """Save current session to disk."""
        if self.current_session and self.current_logger:
            try:
                session_file = os.path.join(self.current_logger.get_run_dir(), "session.json")
                self.current_session.save_to_file(session_file)
            except Exception as e:
                print(f"[App] Warning: Failed to save session: {e}")
    
    def reset_session(self):
        """Reset the current session."""
        try:
            # Finalize current logger
            if self.current_logger:
                self.current_logger.log_step("session_reset", {"status": "reset"})
                self.current_logger.finalize()
                self.current_logger = None
            
            # Reset session
            if self.current_session:
                self.current_session.reset()
            
            self.current_session = None
            self.agent = None
            # Force re-init on next Answer click even if prompt unchanged
            self.current_prompt_system = None
            
            status = "Ready. Click 'Answer' to start a session."
            log_entry = ""
            
            return None, None, None, status, log_entry
            
        except Exception as e:
            error_msg = f"❌ Error resetting session: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return None, error_msg, log_entry
    
    def process_input(self, audio_input: str, image_input: str, prompt_system: str):
        """Process user input (audio and/or image) and generate response."""
        if not audio_input and not image_input:
            return None, "❌ Please provide audio or image input.", "No input provided."
        
        # Auto-start or re-init session if needed
        if (not hasattr(self, 'current_prompt_system')) or (self.current_session is None) or (self.current_prompt_system != prompt_system):
            try:
                self._init_session(prompt_system, with_greeting=False)
            except Exception as e:
                return None, f"❌ Failed to start session: {str(e)}", f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
        
        try:
            # Determine input type and process accordingly
            if audio_input and image_input:
                # Both audio and image provided - process both
                return self._process_audio_and_image(audio_input, image_input)
            elif audio_input:
                # Only audio provided
                return self._process_audio_only(audio_input)
            elif image_input:
                # Only image provided
                return self._process_image_only(image_input)
                
        except Exception as e:
            error_msg = f"❌ Error processing input: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return None, error_msg, log_entry

    def on_prompt_change(self, prompt_system: str):
        """Handle prompt system changes by resetting app state and clearing UI."""
        # Reset state
        try:
            # Finalize logger if present
            if self.current_logger:
                try:
                    self.current_logger.log_step("prompt_change", {"new_prompt": prompt_system})
                except Exception:
                    pass
                self.current_logger.finalize()
        except Exception:
            pass
        self.current_logger = None
        self.current_session = None
        self.agent = None
        # Force re-init on next Answer
        self.current_prompt_system = None
        status = "Ready. Click 'Answer' to start a session."
        log_entry = ""
        # Clear all I/O
        return None, None, None, log_entry

    def _list_sessions(self) -> list:
        """List available session directories in runs dir."""
        try:
            dirs = [d for d in os.listdir(self.runs_dir) if os.path.isdir(os.path.join(self.runs_dir, d))]
            # Sort by modification time, newest first
            dirs.sort(key=lambda d: os.path.getmtime(os.path.join(self.runs_dir, d)), reverse=True)
            return dirs
        except Exception:
            return []

    def refresh_sessions(self):
        """Refresh the sessions dropdown choices."""
        choices = self._list_sessions()
        return gr.Dropdown(choices=choices, value=None)

    def load_session_and_display(self, session_name: Optional[str]):
        """Load a session and display the first interaction."""
        if not session_name:
            return (
                "❌ Please select a session to load.",
                "",
                None,
                "",
                None
            )
        
        session_dir = os.path.join(self.runs_dir, session_name)
        if not os.path.isdir(session_dir):
            return (
                f"❌ Session '{session_name}' not found.",
                "",
                None,
                "",
                None
            )
        
        # Try to load session data from session.json
        session_file = os.path.join(session_dir, "session.json")
        if os.path.exists(session_file):
            try:
                import json
                with open(session_file, 'r', encoding='utf-8') as f:
                    self.loaded_session_data = json.load(f)
            except Exception as e:
                return (
                    f"❌ Failed to load session: {str(e)}",
                    "",
                    None,
                    "",
                    None
                )
        else:
            return (
                f"❌ Session file not found in {session_dir}",
                "",
                None,
                "",
                None
            )
        
        # Parse conversation into user-model interaction pairs
        conversation_history = self.loaded_session_data.get("conversation_history", []) if self.loaded_session_data else []
        self.interaction_pairs = self._parse_interactions(conversation_history)
        
        if not self.interaction_pairs:
            return (
                "📂 Session loaded but no interactions found.",
                "",
                None,
                "",
                None
            )
        
        # Start at first interaction
        self.current_interaction_index = 0
        return self._display_interaction(0)
    
    def _parse_interactions(self, conversation_history: list) -> list:
        """Parse conversation history into user-model interaction pairs."""
        interactions: list = []
        current_pair = {"user": None, "assistant": None}
        
        for message in conversation_history:
            role = message.get("role", "")
            if role == "user":
                # Start new pair if we already have a complete one
                if current_pair["user"] is not None and current_pair["assistant"] is not None:
                    interactions.append(current_pair)
                    current_pair = {"user": None, "assistant": None}
                current_pair["user"] = message
            elif role == "assistant":
                current_pair["assistant"] = message
                # Complete the pair
                if current_pair["user"] is not None:
                    interactions.append(current_pair)
                    current_pair = {"user": None, "assistant": None}
        
        # Add final incomplete pair if exists
        if current_pair["user"] is not None or current_pair["assistant"] is not None:
            interactions.append(current_pair)
        
        return interactions
    
    def _display_interaction(self, index: int):
        """Display a specific interaction by index."""
        if not self.interaction_pairs or index < 0 or index >= len(self.interaction_pairs):
            return (
                "No interaction to display",
                "",
                None,
                "",
                None
            )
        
        pair = self.interaction_pairs[index]
        total = len(self.interaction_pairs)
        
        # Counter
        counter = f"### Interaction {index + 1} / {total}"
        
        # User data
        user_msg = pair.get("user")
        user_transcript = user_msg.get("content", "") if user_msg else ""
        user_metadata = user_msg.get("metadata", {}) if user_msg else {}
        user_image_path = user_metadata.get("image_path")
        
        # Model data
        assistant_msg = pair.get("assistant")
        model_response = assistant_msg.get("content", "") if assistant_msg else ""
        assistant_metadata = assistant_msg.get("metadata", {}) if assistant_msg else {}
        model_audio_path = assistant_metadata.get("audio_path")
        
        return (
            counter,
            user_transcript,
            user_image_path,
            model_response,
            model_audio_path
        )
    
    def navigate_previous(self):
        """Navigate to previous interaction."""
        if not self.interaction_pairs:
            return (
                "No session loaded",
                "",
                None,
                "",
                None
            )
        
        self.current_interaction_index = max(0, self.current_interaction_index - 1)
        return self._display_interaction(self.current_interaction_index)
    
    def navigate_next(self):
        """Navigate to next interaction."""
        if not self.interaction_pairs:
            return (
                "No session loaded",
                "",
                None,
                "",
                None
            )
        
        self.current_interaction_index = min(
            len(self.interaction_pairs) - 1,
            self.current_interaction_index + 1
        )
        return self._display_interaction(self.current_interaction_index)

    def load_session(self, session_name: Optional[str]):
        """Load a session by name; sets state to use its folder for artifacts/log viewing."""
        if not session_name:
            return "❌ Please select a session to load.", "No session selected."
        session_dir = os.path.join(self.runs_dir, session_name)
        if not os.path.isdir(session_dir):
            return f"❌ Session '{session_name}' not found.", f"[{datetime.now().strftime('%H:%M:%S')}] Invalid session selected\n"
        # Note: We don't fully restore conversation; we mark current run dir for reference
        self.loaded_session_dir = session_dir  # type: ignore[attr-defined]
        status = f"📂 Loaded session: {session_name}"
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Session loaded from {session_dir}\n"
        return status, log_entry
    
    def _process_audio_only(self, audio_input: str):
        """Process audio input only."""
        # Process audio with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", [], "Agent not initialized."
        
        result = self.agent.process_input(audio_input, "", {
            "conversation_history": self.current_session.get_conversation_context() if self.current_session else [],
            "last_transcription": ""
        })
        
        if result["success"]:
            # Add user message to session
            transcription = result.get("transcription", "")
            if transcription and self.current_session:
                self.current_session.add_message(
                    role="user",
                    content=transcription,
                    metadata={"audio_path": audio_input}
                )
            
            # Add agent response to session
            if self.current_session:
                response_text = result.get("response_text", "")
                if response_text:  # Add response text if available
                    metadata = {}
                    if result.get("audio_path"):
                        metadata["audio_path"] = result["audio_path"]
                    
                    self.current_session.add_message(
                        role="assistant",
                        content=response_text,
                        metadata=metadata
                    )
            
            # Update execution log with actual console outputs
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Audio processed successfully\n"
            if transcription:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Transcription: '{transcription}'\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ No transcription available\n"
            
            response_text = result.get("response_text", "")
            if response_text:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Agent response: '{response_text}'\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ No response text available\n"
            
            audio_path = result.get("audio_path", "None")
            if audio_path and audio_path != "None":
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] 🔊 Audio generated: {audio_path}\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No audio generated\n"
            
            # Save session to disk
            self._save_current_session()
            
            return result.get("audio_path"), log_entry
        else:
            error_msg = f"❌ Processing failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg
    
    def _process_image_only(self, image_input: str):
        """Process image input only."""
        # Process image with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", [], "Agent not initialized."
        
        result = self.agent.process_input("", image_input, {
            "conversation_history": self.current_session.get_conversation_context() if self.current_session else [],
            "last_transcription": ""
        })
        
        if result["success"]:
            # Add user message to session
            if self.current_session:
                self.current_session.add_message(
                    role="user",
                    content="Image analysis request",
                    metadata={"image_path": image_input}
                )
            
            # Add agent response to session
            if self.current_session:
                response_text = result.get("response_text", "")
                if response_text:  # Add response text if available
                    metadata = {}
                    if result.get("audio_path"):
                        metadata["audio_path"] = result["audio_path"]
                    
                    self.current_session.add_message(
                        role="assistant",
                        content=response_text,
                        metadata=metadata
                    )
            
            # Update execution log with actual console outputs
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Image processed successfully\n"
            response_text = result.get("response_text", "")
            if response_text:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Agent response: '{response_text}'\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ No response text available\n"
            
            audio_path = result.get("audio_path", "None")
            if audio_path and audio_path != "None":
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] 🔊 Audio generated: {audio_path}\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] ❌ No audio generated\n"
            
            # Save session to disk
            self._save_current_session()
            
            return result.get("audio_path"), log_entry
        else:
            error_msg = f"❌ Image analysis failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg
    
    def _process_audio_and_image(self, audio_input: str, image_input: str):
        """Process both audio and image inputs."""
        # Process both audio and image with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", [], "Agent not initialized."
        
        # Create separate context for agent (not Gradio chatbot)
        agent_context = {
            "conversation_history": [],  # Agent maintains its own context
            "last_transcription": ""
        }
        
        result = self.agent.process_input(audio_input, image_input, agent_context)
        
        if result["success"]:
            # Add user message to session
            transcription = result.get("transcription", "")
            user_content = transcription if transcription else "Image analysis request"
            if image_input:
                user_content += " (with image)"
            
            if self.current_session:
                self.current_session.add_message(
                    role="user",
                    content=user_content,
                    metadata={"audio_path": audio_input, "image_path": image_input}
                )
            
            # Add agent response to session
            if self.current_session:
                response_text = result.get("response_text", "")
                if response_text:  # Add response text if available
                    metadata = {}
                    if result.get("audio_path"):
                        metadata["audio_path"] = result["audio_path"]
                    
                    self.current_session.add_message(
                        role="assistant",
                        content=response_text,
                        metadata=metadata
                    )
            
            # Update execution log with actual console outputs
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Multimodal processing\n"
            if transcription:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Audio transcription: '{transcription}'\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] No audio transcription\n"
            
            response_text = result.get("response_text", "")
            if response_text:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Agent response: '{response_text}'\n"
            else:
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] No response text available\n"
            
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Audio generated: {result.get('audio_path', 'None')}\n"
            
            # Save session to disk
            self._save_current_session()
            
            return result.get("audio_path"), log_entry
        else:
            error_msg = f"❌ Multimodal processing failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg
    
    
    
    # Removed _get_chatbot_history method - no longer needed
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.build_interface()
        kwargs.setdefault('show_api', False)
        demo.launch(**kwargs)


def main():
    """Main entry point for the application."""
    
    # Get API key from environment
    api_key = os.getenv("BOSON_API_KEY")
    
    if not api_key:
        print("Warning: BOSON_API_KEY not found in environment variables.")
        print("Please set it using: export BOSON_API_KEY='your-api-key'")
    
    # Set up runs directory
    project_root = os.path.dirname(current_dir)
    runs_dir = os.path.join(project_root, "runs")
    
    print(f"Runs directory: {runs_dir}")
    
    # Create and launch app
    app = VoiceSightApp(api_key=api_key, runs_dir=runs_dir)
    app.launch(
        server_name="localhost",
        server_port=7861,
        share=False
    )

if __name__ == "__main__":
    main()