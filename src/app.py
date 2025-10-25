"""
Voice Sight - Audio-based Agentic Pipeline with Gradio Interface.
"""

import os
import sys
from typing import Optional, Dict, Any
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
    
from utils.voice_sight.agent import VoiceSightAgent  # type: ignore
from utils.voice_sight.session import VoiceSightSession  # type: ignore
from utils.logger import RunLogger, create_logger  # type: ignore


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
        
        # Initialize agent (will be created with logger in start_session)
        self.agent = None
    
    def build_interface(self):
        """Build the Gradio interface."""
        
        with gr.Blocks(title="Voice Sight - Audio Agentic Pipeline", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎙️ Voice Sight - Audio Agentic Pipeline")
            gr.Markdown("An intelligent audio interface that understands and responds through speech.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Audio Input")
                    
                    # Audio input
                    audio_input = gr.Audio(
                        label="Record Your Message",
                        type="filepath",
                        sources=["microphone"],
                        format="wav"
                    )
                    
                    # Image input for visual analysis
                    image_input = gr.Image(
                        label="Upload Image for Visual Analysis",
                        type="filepath",
                        sources=["upload", "webcam"]
                    )
                    
                    # Process button
                    with gr.Row():
                        answer_btn = gr.Button("🎯 Answer", variant="primary")
                    
                    # Session controls
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Session", variant="secondary")
                        reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🔊 Audio Output")
                    
                    # Audio output
                    audio_output = gr.Audio(
                        label="Agent Response",
                        type="filepath",
                        interactive=False
                    )
                    
                    # Status display
                    status_display = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                        value="Ready to start. Click 'Start Session' to begin."
                    )
            
            # Conversation log
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📋 Conversation Log")
                    conversation_log = gr.Textbox(
                        label="Conversation History",
                        interactive=False,
                        lines=8,
                        value="No conversation yet."
                    )
            
            # Execution log
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 Execution Log")
                    execution_log = gr.Textbox(
                        label="System Log",
                        interactive=False,
                        lines=6,
                        value="System ready."
                    )
            
            # Store components for event handlers
            self.components = {
                'audio_input': audio_input,
                'image_input': image_input,
                'audio_output': audio_output,
                'status_display': status_display,
                'conversation_log': conversation_log,
                'execution_log': execution_log
            }
            
            # Event handlers
            start_btn.click(
                fn=self.start_session,
                inputs=[],
                outputs=[status_display, conversation_log, execution_log]
            )
            
            reset_btn.click(
                fn=self.reset_session,
                inputs=[],
                outputs=[audio_output, status_display, conversation_log, execution_log]
            )
            
            answer_btn.click(
                fn=self.process_input,
                inputs=[audio_input, image_input],
                outputs=[audio_output, status_display, conversation_log, execution_log]
            )
        
        return demo
    
    def start_session(self):
        """Start a new Voice Sight session."""
        try:
            # Create new logger
            run_name = f"voice_sight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            self.current_logger = create_logger(run_name=run_name, base_dir=self.runs_dir)
            
            # Create new session
            self.current_session = VoiceSightSession()
            
            # Create agent with logger
            self.agent = VoiceSightAgent(api_key=self.api_key, use_thinking=True, logger=self.current_logger)
            
            # Log session start
            self.current_logger.log_step("session_start", {
                "session_id": self.current_session.session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Start conversation with greeting
            greeting_result = self.agent.start_conversation()
            
            if greeting_result["success"]:
                # Add greeting to session
                self.current_session.add_message(
                    role="assistant",
                    content=greeting_result["text"],
                    metadata={"audio_path": greeting_result["audio_path"]}
                )
                
                status = "✅ Session started! Agent is ready to help."
                conversation = f"🤖 Agent: {greeting_result['text']}"
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Session started (ID: {self.current_session.session_id})\n"
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Greeting generated\n"
                
                return status, conversation, log_entry
            else:
                error_msg = f"❌ Failed to start session: {greeting_result.get('error', 'Unknown error')}"
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {greeting_result.get('error', 'Unknown error')}\n"
                return error_msg, "No conversation yet.", log_entry
                
        except Exception as e:
            error_msg = f"❌ Error starting session: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return error_msg, "No conversation yet.", log_entry
    
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
            
            status = "🔄 Session reset. Click 'Start Session' to begin a new conversation."
            conversation = "No conversation yet."
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Session reset\n"
            
            return None, status, conversation, log_entry
            
        except Exception as e:
            error_msg = f"❌ Error resetting session: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return None, error_msg, "No conversation yet.", log_entry
    
    def process_input(self, audio_input: str, image_input: str):
        """Process user input (audio and/or image) and generate response."""
        if not audio_input and not image_input:
            return None, "❌ Please provide audio or image input.", "No conversation yet.", "No input provided."
        
        if not self.current_session:
            return None, "❌ Please start a session first.", "No conversation yet.", "No active session."
        
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
            return None, error_msg, self._format_conversation_log(), log_entry
    
    def _process_audio_only(self, audio_input: str):
        """Process audio input only."""
        # Process audio with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", "No conversation yet.", "Agent not initialized."
        
        result = self.agent.process_input(audio_input, "", {
            "conversation_history": self.current_session.get_conversation_context(),
            "last_transcription": ""
        })
        
        if result["success"]:
            # Add user message to session
            transcription = result.get("transcription", "")
            if transcription:
                self.current_session.add_message(
                    role="user",
                    content=transcription,
                    metadata={"audio_path": audio_input}
                )
            
            # Add agent response to session
            if result.get("audio_path"):
                response_text = result.get("response_text", "")
                self.current_session.add_message(
                    role="assistant",
                    content=response_text,
                    metadata={"audio_path": result["audio_path"]}
                )
            
            # Update conversation log
            conversation = self._format_conversation_log()
            
            # Update status
            status = "✅ Audio processed successfully!"
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Audio processed\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Transcription: '{transcription}'\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Response generated\n"
            
            
            return result.get("audio_path"), status, conversation, log_entry
        else:
            error_msg = f"❌ Processing failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg, self._format_conversation_log(), log_entry
    
    def _process_image_only(self, image_input: str):
        """Process image input only."""
        # Process image with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", "No conversation yet.", "Agent not initialized."
        
        result = self.agent.process_input("", image_input, {
            "conversation_history": self.current_session.get_conversation_context(),
            "last_transcription": ""
        })
        
        if result["success"]:
            # Add user message to session
            if self.current_session:
                self.current_session.add_message(
                    role="user",
                    content=f"Image analysis request: {image_input}",
                    metadata={"image_path": image_input}
                )
            
            # Add agent response to session
            if result.get("audio_path") and self.current_session:
                response_text = result.get("response_text", "")
                self.current_session.add_message(
                    role="assistant",
                    content=response_text,
                    metadata={"audio_path": result["audio_path"]}
                )
            
            # Update conversation log
            conversation = self._format_conversation_log()
            
            # Update status
            status = "✅ Image analyzed successfully!"
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Image processed\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Analysis: {result.get('analysis', '')}\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Response generated\n"
            
            
            return result.get("audio_path"), status, conversation, log_entry
        else:
            error_msg = f"❌ Image analysis failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg, self._format_conversation_log(), log_entry
    
    def _process_audio_and_image(self, audio_input: str, image_input: str):
        """Process both audio and image inputs."""
        # Process both audio and image with agent
        if not self.agent:
            return None, "❌ Agent not initialized. Please start a session first.", "No conversation yet.", "Agent not initialized."
        
        result = self.agent.process_input(audio_input, image_input, {
            "conversation_history": self.current_session.get_conversation_context(),
            "last_transcription": ""
        })
        
        if result["success"]:
            # Add user message to session
            transcription = result.get("transcription", "")
            self.current_session.add_message(
                role="user",
                content=f"Audio: {transcription} | Image: {image_input}",
                metadata={"audio_path": audio_input, "image_path": image_input}
            )
            
            # Add agent response to session
            if result.get("audio_path"):
                response_text = result.get("response_text", "")
                self.current_session.add_message(
                    role="assistant",
                    content=response_text,
                    metadata={"audio_path": result["audio_path"]}
                )
            
            # Update conversation log
            conversation = self._format_conversation_log()
            
            # Update status
            status = "✅ Audio and image processed successfully!"
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Multimodal processing\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Audio transcription: '{transcription}'\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Image analysis: {result.get('analysis', '')}\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Response generated\n"
            
            
            return result.get("audio_path"), status, conversation, log_entry
        else:
            error_msg = f"❌ Multimodal processing failed: {result.get('error', 'Unknown error')}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {result.get('error', 'Unknown error')}\n"
            return None, error_msg, self._format_conversation_log(), log_entry
    
    
    
    def _format_conversation_log(self) -> str:
        """Format conversation history for display."""
        if not self.current_session:
            return "No conversation yet."
        
        conversation_history = self.current_session.get_conversation_context()
        if not conversation_history:
            return "No conversation yet."
        
        formatted_log = []
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            # timestamp = message["timestamp"]  # Not used in current implementation
            
            if role == "user":
                formatted_log.append(f"👤 You: {content}")
            elif role == "assistant":
                formatted_log.append(f"🤖 Agent: {content}")
        
        return "\n".join(formatted_log)
    
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
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )

if __name__ == "__main__":
    main()