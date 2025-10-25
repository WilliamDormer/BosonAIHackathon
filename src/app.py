"""
Audio-based Agentic Pipeline with Gradio Interface.

Supports multiple functionalities:
1. Translation - Two-way audio translation between Chinese and English
2. [Future] Additional features to be added
"""

import os
import sys
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

from utils.voice_sight.translation import TranslationPipeline, TranslationSession  # type: ignore
from utils.logger import RunLogger, create_logger  # type: ignore


class AudioAgenticApp:
    """Main application class for audio-based agentic pipeline."""
    
    def __init__(self, api_key: Optional[str] = None, runs_dir: str = "runs"):
        """Initialize the application with necessary models."""
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.runs_dir = runs_dir
        
        # Ensure runs directory exists
        os.makedirs(runs_dir, exist_ok=True)
        
        # Current logger (will be created per session)
        self.current_logger: Optional[RunLogger] = None
        
        # Initialize translation pipeline (without logger for now)
        self.translation_pipeline = TranslationPipeline(api_key=self.api_key)
        self.translation_session = TranslationSession(self.translation_pipeline)
    
    def build_interface(self):
        """Build the Gradio interface with multiple tabs."""
        
        with gr.Blocks(title="Audio Agentic Pipeline", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎙️ Audio-Based Agentic Pipeline")
            gr.Markdown("An intelligent audio interface supporting translation and more.")
            
            with gr.Tabs():
                # Tab 1: Translation
                with gr.TabItem("🌐 Translation"):
                    self._build_translation_tab()
                
                # Tab 2: Placeholder for future functionality
                with gr.TabItem("🔮 Feature 2 (Coming Soon)"):
                    gr.Markdown("### Feature 2")
                    gr.Markdown("This feature will be implemented soon.")
                
                # Tab 3: Placeholder for future functionality
                with gr.TabItem("🚀 Feature 3 (Coming Soon)"):
                    gr.Markdown("### Feature 3")
                    gr.Markdown("This feature will be implemented soon.")
        
        return demo
    
    def _build_translation_tab(self):
        """Build the translation tab interface."""
        
        gr.Markdown("## Two-Way Audio Translation")
        gr.Markdown("Translate speech between Chinese and English in real-time.")
        
        # Walkthrough component for step-by-step workflow
        with gr.Walkthrough(selected=0) as walkthrough:
            with gr.Step("📝 Instructions", id=0):
                instructions = gr.Markdown("""
                **Step-by-Step Translation Process:**
                1. Click **Start Session** to begin
                2. **Step 1**: Specify your language preferences (e.g., "Chinese to English")
                3. **Step 2**: Confirm your language choice by saying "confirm"
                4. **Step 3**: Speak in your source language to get translation
                5. The system will automatically progress through each step
                """)
                start_btn = gr.Button("🎬 Start Translation Session", variant="primary", size="lg")
            
            # Step 1: Language Selection
            with gr.Step("🎤 Step 1: Specify Languages", id=1):
                with gr.Column():
                    greeting_audio = gr.Audio(
                        label="Agent Greeting",
                        type="filepath",
                        interactive=False
                    )
                    
                    language_input = gr.Audio(
                        label="Your Response (Specify Languages)",
                        type="filepath",
                        sources=["microphone"]
                    )
                    
                    submit_languages_btn = gr.Button("Submit Language Choice", variant="primary")
            
            # Step 2: Confirmation
            with gr.Step("✅ Step 2: Confirm", id=2):
                with gr.Column():
                    confirmation_audio = gr.Audio(
                        label="Confirmation Prompt",
                        type="filepath",
                        interactive=False
                    )
                    
                    confirmation_input = gr.Audio(
                        label="Say 'Confirm'",
                        type="filepath",
                        sources=["microphone"]
                    )
                    
                    submit_confirmation_btn = gr.Button("Submit Confirmation", variant="primary")
            
            # Step 3: Translation
            with gr.Step("🔊 Step 3: Translate", id=3):
                with gr.Column():
                    translation_input = gr.Audio(
                        label="Speak in Source Language",
                        type="filepath",
                        sources=["microphone"]
                    )
                    
                    translate_btn = gr.Button("🌍 Translate", variant="primary", size="lg")
                    
                    transcribed_text = gr.Textbox(
                        label="Transcribed Text (Original)",
                        interactive=False,
                        lines=3
                    )
                    
                    translated_text = gr.Textbox(
                        label="Translated Text",
                        interactive=False,
                        lines=3
                    )
                    
                    translation_output = gr.Audio(
                        label="Translation (Audio Output)",
                        type="filepath",
                        interactive=False
                    )
        
        reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

        # Execution log (always visible at bottom)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Execution Log")
                execution_log = gr.Textbox(
                    label="Current Session Log",
                    interactive=False,
                    lines=8,
                    value="Session not started yet."
                )
        
        # Store components for access in event handlers
        self.components = {
            'walkthrough': walkthrough,
            'greeting_audio': greeting_audio,
            'language_input': language_input,
            'confirmation_audio': confirmation_audio,
            'confirmation_input': confirmation_input,
            'translation_input': translation_input,
            'transcribed_text': transcribed_text,
            'translated_text': translated_text,
            'translation_output': translation_output
        }
        
        self.shared_components = {
            'execution_log': execution_log,
            'instructions': instructions
        }
        
        # Event handlers
        start_btn.click(
            fn=self.start_translation_session,
            inputs=[],
            outputs=[
                greeting_audio, execution_log, walkthrough
            ]
        )
        
        reset_btn.click(
            fn=self.reset_translation_session,
            inputs=[],
            outputs=[
                greeting_audio, confirmation_audio, transcribed_text, translated_text, translation_output,
                execution_log, walkthrough
            ]
        )
        
        submit_languages_btn.click(
            fn=self.process_language_selection,
            inputs=[language_input],
            outputs=[confirmation_audio, execution_log, walkthrough]
        )
        
        submit_confirmation_btn.click(
            fn=self.process_confirmation,
            inputs=[confirmation_input],
            outputs=[execution_log, walkthrough]
        )
        
        translate_btn.click(
            fn=self.process_translation,
            inputs=[translation_input],
            outputs=[transcribed_text, translated_text, translation_output, execution_log, walkthrough]
        )
        
    
    
    def start_translation_session(self):
        """Start a new translation session with logger."""
        try:
            # Create new logger for this session
            run_name = f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            self.current_logger = create_logger(run_name=run_name, base_dir=self.runs_dir)
            
            # Update pipeline with logger
            self.translation_pipeline.logger = self.current_logger
            
            # Log session start
            self.current_logger.log_step("session_initialization", {
                "feature": "translation",
                "run_name": run_name
            })
            
            greeting_audio_path = self.translation_session.start_session()
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Session started (Run: {run_name})\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Agent greeting generated\n"
            
            return greeting_audio_path, log_entry, gr.Walkthrough(selected=1)
                
        except Exception as e:
            if self.current_logger:
                self.current_logger.log_error("session_start_failed", str(e))
            error_msg = f"❌ Error starting session: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return None, log_entry, gr.Walkthrough(selected=0)
    
    def reset_translation_session(self):
        """Reset the translation session and finalize logger."""
        # Finalize current logger if exists
        if self.current_logger:
            self.current_logger.log_step("session_reset", {"status": "reset"})
            self.current_logger.finalize()
            self.current_logger = None
        
        # Reset session
        self.translation_session.reset()
        
        # Remove logger from pipeline
        self.translation_pipeline.logger = None
        
        status = "Session reset. Click 'Start Translation Session' to begin."
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Session reset\n"
        
        return None, None, "", "", "", None, log_entry, gr.Walkthrough(selected=0)
    
    def process_language_selection(self, audio_input: str):
        """Process user's language selection."""
        if not audio_input:
            return None, ""
        
        if self.translation_session.state != "waiting_languages":
            return None, ""
        
        try:
            if self.current_logger:
                self.current_logger.log_step("language_selection_received", {
                    "audio_file": os.path.basename(audio_input)
                })
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Processing language selection...\n"
            
            # Transcribe user's input
            transcribed = self.translation_pipeline.transcribe_audio(audio_input)
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Transcribed: '{transcribed}'\n"
            
            # Parse language intent
            from_lang, to_lang = self.translation_pipeline.parse_language_intent(transcribed)
            
            if from_lang is None or to_lang is None:
                if self.current_logger:
                    self.current_logger.log_error("language_intent_unclear", f"Could not parse: {transcribed}")
                status = f"❌ Could not understand language preferences. You said: '{transcribed}'\nPlease try again and clearly state both languages."
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Error: Could not parse language intent\n"
                return None, status, log_entry
            
            # Store languages
            self.translation_session.from_lang = from_lang
            self.translation_session.to_lang = to_lang
            self.translation_session.state = "waiting_confirmation"
            
            if self.current_logger:
                self.current_logger.log_result("language_selection", {
                    "from_language": from_lang,
                    "to_language": to_lang,
                    "user_input": transcribed
                })
            
            # Generate confirmation prompt
            confirmation_audio = self.translation_pipeline.generate_confirmation_prompt(
                from_lang, to_lang
            )
            
            lang_names = self.translation_pipeline.language_names
            status = f"✅ Understood: {lang_names[from_lang]} → {lang_names[to_lang]}\nPlease listen and confirm."
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Languages selected: {lang_names[from_lang]} → {lang_names[to_lang]}\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Confirmation prompt generated\n"
            
            return confirmation_audio, log_entry, gr.Walkthrough(selected=2)
            
        except Exception as e:
            if self.current_logger:
                self.current_logger.log_error("language_selection_error", str(e))
            error_msg = f"❌ Error processing language selection: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return None, log_entry, gr.Walkthrough(selected=1)
    
    def process_confirmation(self, audio_input: str):
        """Process user's confirmation."""
        if not audio_input:
            return "", gr.Walkthrough(selected=2)
        
        if self.translation_session.state != "waiting_confirmation":
            return "", gr.Walkthrough(selected=2)
        
        try:
            if self.current_logger:
                self.current_logger.log_step("confirmation_received", {
                    "audio_file": os.path.basename(audio_input)
                })
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Processing confirmation...\n"
            
            # Check if user confirmed
            confirmed = self.translation_pipeline.check_confirmation(audio_input)
            
            if confirmed:
                self.translation_session.state = "translating"
                
                if self.current_logger:
                    self.current_logger.log_result("confirmation", {
                        "confirmed": True,
                        "state": "translating"
                    })
                
                # TODO: let the agent speak this message
                status = "Confirmed! You can now speak in your source language and click 'Translate'."
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Confirmation successful\n"
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Ready for translation\n"
                
                return log_entry, gr.Walkthrough(selected=3)
            else:
                transcribed = self.translation_pipeline.transcribe_audio(audio_input)
                
                if self.current_logger:
                    self.current_logger.log_warning("confirmation_failed", f"User said: {transcribed}")
                
                # TODO: let the agent speak this message
                status = f"Not confirmed. You said: '{transcribed}'\nPlease say 'confirm' to proceed."
                
                log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Confirmation failed: '{transcribed}'\n"
                
                return log_entry, gr.Walkthrough(selected=2)
                
        except Exception as e:
            if self.current_logger:
                self.current_logger.log_error("confirmation_error", str(e))
            error_msg = f"❌ Error processing confirmation: {str(e)}"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}\n"
            return log_entry, gr.Walkthrough(selected=2)
    
    def process_translation(self, audio_input: str):
        """Process audio translation."""
        if not audio_input:
            return "❌ Please record audio to translate.", "", None, ""
        
        if self.translation_session.state != "translating":
            return "❌ Please complete the setup steps first.", "", None, ""
        
        try:
            # Check that languages are set
            if not self.translation_session.from_lang or not self.translation_session.to_lang:
                return "❌ Language settings missing. Please restart the session.", "", None, ""
            
            if self.current_logger:
                self.current_logger.log_step("translation_received", {
                    "audio_file": os.path.basename(audio_input),
                    "from_language": self.translation_session.from_lang,
                    "to_language": self.translation_session.to_lang
                })
            
            # Update execution log
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Starting translation...\n"
            
            # Perform audio-to-audio translation
            transcribed, translated, output_audio = self.translation_pipeline.translate_audio_to_audio(
                input_audio_path=audio_input,
                from_lang=self.translation_session.from_lang,
                to_lang=self.translation_session.to_lang
            )
            
            # Log final results
            if self.current_logger:
                self.current_logger.log_result("translation_complete", {
                    "from_language": self.translation_session.from_lang,
                    "to_language": self.translation_session.to_lang,
                    "original_text": transcribed,
                    "translated_text": translated,
                    "output_audio_file": os.path.basename(output_audio)
                })
                
                # Save a copy of output audio to run directory
                try:
                    import shutil
                    audio_copy = self.current_logger.run_dir + "/translation_output.wav"
                    shutil.copy(output_audio, audio_copy)
                except Exception:
                    pass
            
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Translation complete\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Original: '{transcribed}'\n"
            log_entry += f"[{datetime.now().strftime('%H:%M:%S')}] Translated: '{translated}'\n"
            
            return transcribed, translated, output_audio, log_entry, gr.Walkthrough(selected=3)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.current_logger:
                self.current_logger.log_error("translation_error", error_msg)
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Error: {error_msg}\n"
            return f"❌ Error during translation: {error_msg}", "", None, log_entry
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.build_interface()
        # Disable API info to avoid schema parsing issues
        kwargs.setdefault('show_api', False)
        demo.launch(**kwargs)


def main():
    """Main entry point for the application."""
    
    # Get API key from environment
    api_key = os.getenv("BOSON_API_KEY")
    
    if not api_key:
        print("Warning: BOSON_API_KEY not found in environment variables.")
        print("Please set it using: export BOSON_API_KEY='your-api-key'")
    
    # Set up runs directory (relative to project root)
    project_root = os.path.dirname(current_dir)
    runs_dir = os.path.join(project_root, "runs")
    
    print(f"Runs directory: {runs_dir}")
    
    # Create and launch app
    app = AudioAgenticApp(api_key=api_key, runs_dir=runs_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()

