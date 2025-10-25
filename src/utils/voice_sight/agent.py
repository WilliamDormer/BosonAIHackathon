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
        
        # Image availability tracking
        self.has_image = False
        self.current_image_path: Optional[str] = None
        
        # Tool definitions for LLM
        self.tools = self._define_tools()
        
        # System prompt
        self.system_prompt = self._get_system_prompt()
    
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
                    },
                    "required": ["text"]
                }
            ),
            create_tool_call_schema(
                tool_name="visual_analysis",
                description="Analyze the uploaded image for the user - use this when you need to provide visual assistance",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question about the image analysis",
                            "default": "Analyze this image for a visually-impaired person, focusing on safety and navigation."
                        }
                    },
                    "required": []
                }
            )
        ]
    
    def register_image(self, image_path: str) -> None:
        """
        Register an image for the current session.
        
        Args:
            image_path: Path to the uploaded image
        """
        self.has_image = True
        self.current_image_path = image_path
        
        # Update system prompt to include image availability
        self.system_prompt = self._get_system_prompt()
        
        if self.logger:
            self.logger.log_step("image_registered", {
                "image_path": image_path,
                "has_image": self.has_image
            })
    
    def clear_image(self) -> None:
        """Clear the current image from the session."""
        self.has_image = False
        self.current_image_path = None
        
        # Update system prompt to remove image availability
        self.system_prompt = self._get_system_prompt()
        
        if self.logger:
            self.logger.log_step("image_cleared", {
                "has_image": self.has_image
            })
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt, updated based on image availability."""
        base_prompt = self.prompts.get("system_prompts", {}).get("main_agent", "")
        
        if self.has_image:
            image_context = "\n\n## Current Session Context:\nYou have access to an uploaded image that the user has provided. You can analyze this image using the visual analysis tool to help the user with their request. The image is available for analysis whenever you need to provide visual assistance."
            return base_prompt + image_context
        else:
            return base_prompt
    
    def process_input(self, audio_path: str, image_path: str, session_context: Dict[str, Any], test_mode: bool = False) -> Dict[str, Any]:
        """
        Process audio input from user and generate appropriate response.
        
        Args:
            audio_path: Path to user's audio input
            image_path: Path to user's image input
            session_context: Current session context
            
        Returns:
            Dictionary containing response audio path and metadata
        """
        try:            
            if not test_mode:
                # Step 1: Transcribe the audio
                if audio_path:
                    transcription = self._transcribe_audio(audio_path)
                else:
                    transcription = ""
            else:
                transcription = "Test mode: No audio input provided"
                
            if image_path:
                self.register_image(image_path)
            else:
                self.clear_image()
            
            # Step 2: Create conversation context
            # llm has already known there's an image available as self.register_image() add contextual info to system prompt
            messages = self._build_conversation_context(transcription, session_context)
            
            # Step 3: Get LLM response with tool calls
            response = self._get_llm_response(messages)
            
            # Step 4: Execute tool calls with current messages context
            session_context["current_messages"] = messages
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
            # Use ASR to transcribe with appropriate prompt
            asr_prompt = self.prompts.get("tool_prompts", {}).get("asr", "")
            result = self.asr.call(audio=audio_path, system_prompt=asr_prompt, validate_json=True)
            
            # Parse JSON response
            try:
                import json
                parsed_result = json.loads(result)
                transcription = parsed_result.get("response", "")
            except (json.JSONDecodeError, KeyError):
                # Fallback to raw result if JSON parsing fails
                print(f"[ASR] Error parsing JSON response: {result}, treating it as a string")
                transcription = result
            
            # Log user input
            if self.logger:
                logger_text = f"🤫 user\nTranscription: {transcription}\nAudio file: {os.path.basename(audio_path)}"
                logger_payload = {
                    "action": "audio_input",
                    "content": transcription,
                    "audio_file": os.path.basename(audio_path)
                }
                
                self.logger.log_result(logger_text, logger_payload)
            
            return transcription
        except Exception as e:
            if self.logger:
                self.logger.log_error("audio_transcription_failed", str(e))
            print(f"Error transcribing audio: {e}")
            return ""
    
    def _build_conversation_context(self, transcription: str,session_context: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            
            # Add tools to the messages for function calling
            messages_with_tools = messages.copy()
            if self.tools:
                messages_with_tools.append({
                    "role": "system",
                    "content": f"Available tools: {json.dumps(self.tools, indent=2)}"
                })
            
            # Use the LLM.chat_completion method
            response = self.llm.chat_completion(
                assembled_payloads=messages_with_tools,
                validate_json=True
            )
            
            # Parse response and extract content and tool calls
            response_text, tool_calls = self._parse_llm_response(response)
            
            # Log agent response
            if self.logger:
                logger_text = f"🤖 agent\nResponse: {response_text}\nTool calls: {len(tool_calls)}"
                logger_payload = {
                    "action": "llm_response",
                    "content": response_text,
                    "tool_calls_count": len(tool_calls),
                    "thinking_mode": self.llm.use_thinking
                }
                self.logger.log_result(logger_text, logger_payload)
            
            return {
                "content": response_text,
                "tool_calls": tool_calls,
                "role": "assistant"
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("llm_processing_failed", str(e))
            print(f"Error getting LLM response: {e}")
            return {"content": "", "tool_calls": [], "role": "assistant"}
    
    def _parse_llm_response(self, response: str) -> tuple[str, List[Dict[str, Any]]]:
        """Parse LLM response to extract content and tool calls."""
        import re  # type: ignore
        
        try:
            # Extract response text from <response></response> tags
            response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
            response_text = response_match.group(1).strip() if response_match else response
            
            # Extract tool calls from <tool_call></tool_call> tags
            tool_calls = []
            tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
            tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
            
            for tool_call_text in tool_call_matches:
                try:
                    # Parse tool call JSON
                    tool_call_data = json.loads(tool_call_text.strip())
                    tool_calls.append({
                        "function": {
                            "name": tool_call_data.get("name"),
                            "arguments": json.dumps(tool_call_data.get("arguments", {}))
                        }
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[LLM] Error parsing tool call: {tool_call_text}, error: {e}")
                    continue
            
            return response_text, tool_calls
            
        except Exception as e:
            print(f"[LLM] Error parsing response: {response}, error: {e}")
            return response, []
    
    def _execute_tool_calls(self, response: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls from LLM response with iterative calling for visual analysis."""
        tool_calls = response.get("tool_calls", [])
        results = []
        audio_path = None
        current_messages = session_context.get("current_messages", [])
        
        # If there are tool calls, execute them
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    function_name = tool_call.get("function", {}).get("name")
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    
                    if function_name == "generate_audio":
                        result = self._execute_generate_audio(arguments)
                        if result.get("success"):
                            audio_path = result.get("audio_path")
                    elif function_name == "visual_analysis":
                        result = self._execute_visual_analysis(arguments)
                        # If visual analysis succeeds, continue with LLM
                        if result.get("success"):
                            # Add visual analysis result to conversation context
                            current_messages.append({
                                "role": "assistant",
                                "content": f"Visual analysis result: {result.get('analysis', '')}"
                            })
                            
                            # Continue with LLM to get audio generation
                            if self.logger:
                                self.logger.log_result("🤖 agent", {
                                    "action": "follow_up_llm_call",
                                    "content": "Continuing with LLM after visual analysis",
                                    "context": "visual_analysis_complete"
                                })
                            
                            follow_up_response = self._get_llm_response(current_messages)
                            follow_up_tool_calls = follow_up_response.get("tool_calls", [])
                            
                            # Execute follow-up tool calls (should include audio generation)
                            for follow_up_call in follow_up_tool_calls:
                                follow_up_name = follow_up_call.get("function", {}).get("name")
                                follow_up_args = json.loads(follow_up_call.get("function", {}).get("arguments", "{}"))
                                
                                if follow_up_name == "generate_audio":
                                    if self.logger:
                                        self.logger.log_result("🔈 audio", {
                                            "action": "audio_generation",
                                            "content": "Generating audio from follow-up LLM call",
                                            "text": follow_up_args.get("text", "")
                                        })
                                    
                                    audio_result = self._execute_generate_audio(follow_up_args)
                                    if audio_result.get("success"):
                                        audio_path = audio_result.get("audio_path")
                                        results.append({
                                            "function_name": "generate_audio",
                                            "result": audio_result
                                        })
                            
                            # Update response content with follow-up content
                            if follow_up_response.get("content"):
                                response["content"] = follow_up_response.get("content")
                        
                        # Add visual analysis result to results
                        results.append({
                            "function_name": "visual_analysis",
                            "result": result
                        })
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
        
        # Only generate audio if the LLM explicitly calls the generate_audio tool
        # No TTS fallback - the LLM must use the generate_audio tool for audio output
        
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
            
            # Log audio generation
            if self.logger:
                logger_text = f"🔈 audio\nText: {text}\nVoice: {voice}\nAudio path: {output_path}"
                logger_payload = {
                    "action": "audio_generation",
                    "content": text,
                    "voice": voice,
                    "audio_path": output_path
                }
                self.logger.log_result(logger_text, logger_payload)
                
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
    
    def _execute_visual_analysis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visual analysis for images."""
        try:
            if not self.has_image or not self.current_image_path:
                return {
                    "success": False,
                    "error": "No image available for analysis. Please as the user to provide an image to you."
                }
            
            question = arguments.get("question", "Analyze this for a visually-impaired person, focusing on safety and navigation.")
            
            # Use MLLM for image analysis with the registered image
            mllm_prompt = self.prompts.get("tool_prompts", {}).get("mllm", "")
            result = self.mllm.call(
                system_prompt=mllm_prompt,
                user_prompt=question,
                image=self.current_image_path
            )
            
            # Parse JSON response
            try:
                import re
                # first match the <response></response> tags and return the text inside
                response_match = re.search(r'<response>(.*?)</response>', result, re.DOTALL)
                response_text = response_match.group(1).strip() if response_match else result
                analysis = response_text
            except (re.error, AttributeError):
                print(f"[MLLM] Failed to get analysis from response: {result}, treating it as a string")
                analysis = result
            
            # Log visual analysis
            if self.logger:
                logger_text = f"🌇 visual analyzer\nAnalysis: {analysis}\nQuestion: {question}\nImage path: {self.current_image_path}"
                logger_payload = {
                    "action": "visual_analysis",
                    "content": analysis,
                    "question": question,
                    "image_path": self.current_image_path
                }
                self.logger.log_result(logger_text, logger_payload)
                
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("visual_analysis_failed", str(e))
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
    
