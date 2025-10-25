"""
Voice Sight Agent - Main LLM agent with tool calling capabilities for visually-impaired assistance.
"""

import os
import json
import tempfile
import yaml  # type: ignore
from typing import Optional, Dict, Any, List

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.llm import LLM  # type: ignore
from models.asr import ASR  # type: ignore
from models.tts import TTS  # type: ignore
from models.mllm import MLLM  # type: ignore

# Import utilities
from utils.helpers import (  # type: ignore
    create_text_message,
    create_tool_call_schema
)


class VoiceSightAgent:
    """Main agent that orchestrates audio understanding, LLM reasoning, and audio generation for visually-impaired assistance."""
    
    def __init__(self, api_key: Optional[str] = None, use_thinking: bool = True, logger: Optional[Any] = None):
        """
        Initialize the Voice Sight agent.
        
        Args:
            api_key: API key for Boson AI services
            use_thinking: Whether to use thinking model for better reasoning
            logger: Logger instance for tracking operations
        """
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.logger = logger
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Initialize models with config
        self.llm = LLM(use_thinking=use_thinking, api_key_override=self.api_key)
        self.asr = ASR(api_key_override=self.api_key)
        self.tts = TTS(api_key_override=self.api_key)
        self.mllm = MLLM(api_key_override=self.api_key)
        
        # Tool definitions for LLM
        self.tools = self._define_tools()
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        prompts_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "voice_sight.yaml")
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[VoiceSight] Warning: Failed to load prompts: {e}. Using defaults.")
            return {}
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for the LLM."""
        return [
            create_tool_call_schema(
                tool_name="generate_audio",
                description="Generate audio speech from text using TTS - use this for all audio responses to users",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to convert to speech"
                        },
                        "voice": {
                            "type": "string",
                            "description": "Voice to use for generation",
                            "enum": ["belinda", "broom_salesman", "chadwick", "en_man", "en_woman", "mabel", "vex", "zh_man_sichuan"],
                            "default": "en_woman"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated audio file"
                        }
                    },
                    "required": ["text"]
                }
            ),
            create_tool_call_schema(
                tool_name="analyze_multimodal",
                description="General-purpose multimodal analysis tool for images and audio - use this for all visual and audio analysis",
                parameters={
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Path to image or audio file to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform",
                            "enum": ["safety", "navigation", "objects", "scene", "traffic", "obstacles", "audio_analysis"],
                            "default": "safety"
                        },
                        "question": {
                            "type": "string",
                            "description": "Specific question about the input",
                            "default": "Analyze this for a visually-impaired person, focusing on safety and navigation."
                        }
                    },
                    "required": ["input_path"]
                }
            )
        ]
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        system_prompts = self.prompts.get("system_prompts", {})
        return system_prompts.get("main_agent", "")
    
    def process_audio_input(self, audio_path: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio input from user and generate appropriate response.
        
        Args:
            audio_path: Path to user's audio input
            session_context: Current session context
            
        Returns:
            Dictionary containing response audio path and metadata
        """
        try:
            # Step 1: Transcribe the audio
            transcription = self._transcribe_audio(audio_path)
            
            # Step 2: Create conversation context
            messages = self._build_conversation_context(transcription, session_context)
            
            # Step 3: Get LLM response with tool calls
            response = self._get_llm_response(messages)
            
            # Step 4: Execute tool calls
            result = self._execute_tool_calls(response, session_context)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "audio_path": None,
                "transcription": None,
                "response_text": None
            }
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        try:
            if self.logger:
                self.logger.log_step("audio_transcription", {
                    "audio_file": os.path.basename(audio_path)
                })
            
            # Use ASR to transcribe with appropriate prompt
            asr_prompt = self.prompts.get("model_prompts", {}).get("asr_transcription", "Transcribe this audio accurately.")
            result = self.asr.call(audio=audio_path, system_prompt=asr_prompt, validate_json=True)
            
            # Parse JSON response
            try:
                import json
                parsed_result = json.loads(result)
                transcription = parsed_result.get("response", "")
            except (json.JSONDecodeError, KeyError):
                # Fallback to raw result if JSON parsing fails
                transcription = result
            
            if self.logger:
                self.logger.log_result("audio_transcription_complete", {
                    "transcription": transcription,
                    "audio_file": os.path.basename(audio_path)
                })
            
            return transcription
        except Exception as e:
            if self.logger:
                self.logger.log_error("audio_transcription_failed", str(e))
            print(f"Error transcribing audio: {e}")
            return ""
    
    def _build_conversation_context(self, transcription: str, session_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build conversation context for LLM."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add session context if available
        if session_context.get("conversation_history"):
            messages.extend(session_context["conversation_history"][-5:])  # Last 5 messages
        
        # Add current user input
        if transcription:
            messages.append(create_text_message(transcription))
        
        return messages
    
    def _get_llm_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get response from LLM with tool calling."""
        try:
            if self.logger:
                self.logger.log_step("llm_processing", {
                    "message_count": len(messages),
                    "tools_available": len(self.tools)
                })
            
            # Get appropriate LLM prompt based on thinking mode
            llm_prompt = self.prompts.get("model_prompts", {}).get("llm_thinking" if self.llm.use_thinking else "llm_non_thinking", "")
            
            # Add model-specific prompt to system message if available
            if llm_prompt and messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{llm_prompt}"
            
            # Use the LLM.call method instead of chat_completion
            system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
            user_prompt = ""
            
            # Extract user message from the last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break
            
            response = self.llm.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                validate_json=True
            )
            
            # Parse JSON response
            try:
                import json
                parsed_response = json.loads(response)
                content = parsed_response.get("response", response)
            except (json.JSONDecodeError, KeyError):
                content = response
            
            if self.logger:
                self.logger.log_result("llm_response_complete", {
                    "content_length": len(content),
                    "thinking_mode": self.llm.use_thinking
                })
            
            return {
                "content": content,
                "tool_calls": [],  # Simplified - no tool calls for now
                "role": "assistant"
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("llm_processing_failed", str(e))
            print(f"Error getting LLM response: {e}")
            return {"content": "", "tool_calls": [], "role": "assistant"}
    
    def _execute_tool_calls(self, response: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls from LLM response."""
        tool_calls = response.get("tool_calls", [])
        results = []
        audio_path = None
        
        # If there are tool calls, execute them
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    function_name = tool_call.get("function", {}).get("name")
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    
                    if function_name == "generate_audio":
                        result = self._execute_generate_audio(arguments)
                    elif function_name == "analyze_multimodal":
                        result = self._execute_analyze_multimodal(arguments)
                    else:
                        result = {"error": f"Unknown function: {function_name}"}
                    
                    results.append({
                        "function_name": function_name,
                        "result": result
                    })
                    
                except Exception as e:
                    results.append({
                        "function_name": tool_call.get("function", {}).get("name", "unknown"),
                        "result": {"error": str(e)}
                    })
            
            # Find audio generation result from tool calls
            for result in results:
                if result["function_name"] == "generate_audio" and result["result"].get("audio_path"):
                    audio_path = result["result"]["audio_path"]
                    break
        
        # If no audio was generated from tool calls, generate audio from LLM response
        if not audio_path:
            try:
                response_text = response.get("content", "")
                if response_text:
                    # Generate audio from the LLM response
                    audio_result = self._execute_generate_audio({"text": response_text})
                    if audio_result.get("success"):
                        audio_path = audio_result.get("audio_path")
                        results.append({
                            "function_name": "generate_audio",
                            "result": audio_result
                        })
            except Exception as e:
                print(f"Error generating audio from response: {e}")
        
        return {
            "success": True,
            "audio_path": audio_path,
            "transcription": session_context.get("last_transcription", ""),
            "response_text": response.get("content", ""),
            "tool_results": results
        }
    
    def _execute_generate_audio(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio generation."""
        try:
            text = arguments.get("text", "")
            voice = arguments.get("voice", "en_woman")
            
            if self.logger:
                self.logger.log_step("audio_generation", {
                    "text_length": len(text),
                    "voice": voice
                })
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Generate audio with appropriate settings
            self.tts.generate_simple(
                text=text,
                voice=voice,
                output_path=output_path
            )
            
            if self.logger:
                self.logger.log_result("audio_generation_complete", {
                    "audio_path": output_path,
                    "text": text,
                    "voice": voice
                })
            
            return {
                "success": True,
                "audio_path": output_path,
                "text": text,
                "voice": voice
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("audio_generation_failed", str(e))
            return {"success": False, "error": str(e)}
    
    def _execute_analyze_multimodal(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multimodal analysis for images and audio."""
        try:
            input_path = arguments.get("input_path", "")
            analysis_type = arguments.get("analysis_type", "safety")
            question = arguments.get("question", "Analyze this for a visually-impaired person, focusing on safety and navigation.")
            
            # Determine if input is image or audio based on file extension
            file_ext = input_path.lower().split('.')[-1] if '.' in input_path else ""
            is_audio = file_ext in ['wav', 'mp3', 'm4a', 'flac', 'ogg']
            is_image = file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
            
            if is_audio:
                # Use ASR for audio analysis
                system_prompt = "You are an audio analysis assistant. Always respond with JSON format: {\"response\": \"your analysis\"}"
                result = self.asr.call(
                    audio=input_path,
                    system_prompt=system_prompt,
                    user_prompt=question,
                    validate_json=True
                )
            elif is_image:
                # Use MLLM for image analysis
                system_prompt = f"You are a visual analysis assistant for visually-impaired users. Focus on {analysis_type}. Always respond with JSON format: {{\"response\": \"your analysis\"}}"
                result = self.mllm.call(
                    system_prompt=system_prompt,
                    user_prompt=question,
                    image=input_path
                )
            else:
                return {"success": False, "error": f"Unsupported file type: {file_ext}"}
            
            # Parse JSON response
            try:
                import json
                parsed_result = json.loads(result)
                analysis = parsed_result.get("response", "")
            except (json.JSONDecodeError, KeyError):
                analysis = result
            
            return {
                "success": True,
                "analysis": analysis,
                "input_path": input_path,
                "analysis_type": analysis_type,
                "input_type": "audio" if is_audio else "image"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    
    def start_conversation(self) -> Dict[str, Any]:
        """Start a new conversation with greeting."""
        try:
            greeting_text = "Hello! I'm Voice Sight, your audio-based AI assistant. I can understand your speech and respond with audio. How can I help you today?"
            
            # Generate greeting audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
            
            self.tts.generate_simple(
                text=greeting_text,
                voice="en_woman",
                output_path=output_path
            )
            
            return {
                "success": True,
                "audio_path": output_path,
                "text": greeting_text,
                "message": "Conversation started"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "audio_path": None,
                "text": None
            }
    
