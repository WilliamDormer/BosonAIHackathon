"""
Voice Sight Agent - Main LLM agent with tool calling capabilities for visually-impaired assistance.
"""

import os
import time
import json
import tempfile
import yaml  # type: ignore
from typing import Optional, Dict, Any, List

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.llm import LLM  # type: ignore
from models.asr import ASR  # type: ignore
# from models.asr_qwen_omni import ASR
from models.tts import TTS  # type: ignore
from models.mllm import MLLM  # type: ignore

# Import utilities
from utils.helpers import (  # type: ignore
    create_text_message,
    create_tool_call_schema
)


class VoiceSightAgent:
    """Main agent that orchestrates audio understanding, LLM reasoning, and audio generation for visually-impaired assistance."""
    
    def __init__(self, api_key: Optional[str] = None, use_thinking: bool = True, logger: Optional[Any] = None, prompt_system: str = "voice_sight.yaml"):
        """
        Initialize the Voice Sight agent.
        
        Args:
            api_key: API key for Boson AI services
            use_thinking: Whether to use thinking model for better reasoning
            logger: Logger instance for tracking operations
            prompt_system: Name of the prompt system YAML file to use
        """
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.logger = logger
        self.prompt_system = prompt_system
        
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
        
        # Agent's own conversation context (separate from Gradio chatbot)
        self.conversation_context: List[Dict[str, Any]] = []
        
        # Tool definitions for LLM
        self.tools = self._define_tools()
        
        # System prompt
        self.system_prompt = self._get_system_prompt()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        prompts_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", self.prompt_system)
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[VoiceSight] Warning: Failed to load prompts from {self.prompt_system}: {e}. Using defaults.")
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
                            "default": "belinda"
                        },
                    },
                    "required": ["text"]
                }
            ),
            create_tool_call_schema(
                tool_name="visual_analysis",
                description="Analyze the uploaded image for the user - use this when you need to provide visual assistance or get visual cues from the image",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question about the image analysis/perception",
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
    
    def clear_conversation_context(self) -> None:
        """Clear the agent's conversation context."""
        self.conversation_context = []
        
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
            # Reset visual analysis flag for new input
            session_context["visual_analysis_done"] = False
            
            if not test_mode:
                # Step 1: Transcribe the audio
                if audio_path:
                    if self.logger:
                        self.logger.log_result("🎤 ASR: starting transcrption", {
                            "action": "audio_transcription_start",
                            "content": "Starting audio transcription"
                        })
                    transcription = self._transcribe_audio(audio_path)
                    if self.logger:
                        self.logger.log_result("🎤 ASR: transcription complete", {
                            "action": "audio_transcription_complete",
                            "content": f"Transcription: '{transcription}'"
                        })
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
            
            # Store transcription in session context for later use
            session_context["last_transcription"] = transcription
            
            # Step 3: Get LLM response with tool calls
            response = self._get_llm_response(messages)
            
            # Step 4: Execute tool calls with current messages context
            session_context["current_messages"] = messages
            result = self._execute_tool_calls(response, session_context)
            
            # Step 5: Update agent's own conversation context
            self._update_conversation_context(transcription, result)
            
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
                
                # try to retrieve "{""response": "transcription text"}" pattern
                try: 
                    import re
                    response_match = re.search(r'{"response"\s*:\s*"(.*?)"}', result)
                    print(f"[ASR] Fallback regex match result: {response_match}")
                    if response_match:
                        transcription = response_match.group(1)

                except Exception as e:
                    print(f"[ASR] Further error parsing response using back-up mode: {e}")
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
    
    def _build_conversation_context(self, transcription: str, session_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build conversation context for LLM using agent's own context."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Explicit session context about image availability for this turn
        if self.has_image and self.current_image_path:
            messages.append({
                "role": "system",
                "content": (
                    "Current turn context: An image has been uploaded for this turn and is available for analysis. "
                    "If helpful, call the visual_analysis function with a concise question relevant to the user's need."
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "Current turn context: No image is available for this turn. Do not call the visual_analysis function."
                )
            })
        
        # Add agent's own conversation context (last 20 messages for multi-turn)
        if self.conversation_context:
            messages.extend(self.conversation_context[-20:])
        
        # Add current user input
        if transcription:
            messages.append(create_text_message(transcription))
        
        return messages
    
    def _update_conversation_context(self, transcription: str, result: Dict[str, Any]) -> None:
        """Update agent's own conversation context."""
        # Add user message
        if transcription:
            self.conversation_context.append({
                "role": "user",
                "content": transcription
            })
        
        # Add assistant response
        if result.get("success") and result.get("response_text"):
            self.conversation_context.append({
                "role": "assistant", 
                "content": result["response_text"]
            })
    
    def _get_llm_response(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """Get response from LLM with tool calling and error handling."""
        try:
            if self.logger:
                self.logger.log_step("llm_processing", {
                    "message_count": len(messages),
                    "tools_available": len(self.tools)
                })
            
            # Try with different temperatures if no tool calls are generated
            for attempt in range(max_retries):
                # Use temperature != 0.0 for better function calling
                temperature = 0.7 if attempt == 0 else 0.9  # Start with 0.7, increase to 0.9 on retries
                
                if self.logger and attempt > 0:
                    self.logger.log_result("🤖 Agent: Retrying generation with different temperature.", {
                        "action": "llm_retry",
                        "content": f"Retrying LLM call (attempt {attempt + 1}/{max_retries})",
                        "temperature": temperature
                    })
                
                # Use the LLM.chat_completion method with standardized function calling
                response_message = self.llm.chat_completion(
                    assembled_payloads=messages,
                    validate_json=False,  # Don't validate JSON when using function calling
                    temperature=temperature,
                    tools=self.tools,  # Pass tools directly to API
                    tool_choice="auto"  # Let the model decide when to call functions
                )
                # Ensure we can access dynamic attributes from OpenAI message object
                from typing import cast as _cast_any
                response_message = _cast_any(Any, response_message)
                
                # Extract content and tool calls from the response message
                response_text = response_message.content if response_message.content else ""
                tool_calls = response_message.tool_calls if response_message.tool_calls else []
                
                # Convert tool_calls to our expected format
                formatted_tool_calls = []
                if tool_calls:
                    for tool_call in tool_calls:
                        formatted_tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                
                # Check if we have at least 1 function call
                if formatted_tool_calls and len(formatted_tool_calls) > 0:
                    # Success - we have tool calls
                    if self.logger:
                        self.logger.log_result("🤖 Agent: Generated Toolcalls", {
                            "action": "llm_success",
                            "content": f"LLM generated {len(formatted_tool_calls)} tool calls on attempt {attempt + 1}",
                            "temperature": temperature
                        })
                    break
                elif attempt < max_retries - 1:
                    # No tool calls, but we can retry
                    if self.logger:
                        self.logger.log_result("🤖 Agent: No Tool calls!", {
                            "action": "llm_no_tool_calls",
                            "content": f"No tool calls generated, retrying with temperature {temperature}",
                            "attempt": attempt + 1
                        })
                    continue
                else:
                    # Final attempt failed
                    if self.logger:
                        self.logger.log_result("🤖 Agent: No tool calls after maximum retries", {
                            "action": "llm_no_tool_calls_final",
                            "content": f"No tool calls generated after {max_retries} attempts",
                            "temperature": temperature
                        })
            
            # Store the clean response text for later use
            self.last_response_text = response_text
            
            # Log agent response
            if self.logger:
                logger_text = f"🤖 Agent: \nResponse: {response_text}\nTool calls: {len(formatted_tool_calls)}"
                logger_payload = {
                    "action": "llm_response",
                    "content": response_text,
                    "tool_calls_count": len(formatted_tool_calls),
                    "thinking_mode": self.llm.use_thinking
                }
                self.logger.log_result(logger_text, logger_payload)
            
            return {
                "content": response_text,
                "tool_calls": formatted_tool_calls,
                "role": "assistant"
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("llm_processing_failed", str(e))
            print(f"Error getting LLM response: {e}")
            return {"content": "", "tool_calls": [], "role": "assistant"}
    
    def _execute_tool_calls(self, response: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls iteratively until no more tool calls are returned (bounded loop)."""
        results: List[Dict[str, Any]] = []
        audio_path: Optional[str] = None
        current_messages: List[Dict[str, Any]] = session_context.get("current_messages", [])
        
        # Iterative tool execution loop (bounded to prevent runaway)
        max_iterations = 20
        iteration = 0
        current_response = response
        
        while iteration < max_iterations:
            iteration += 1
            tool_calls = current_response.get("tool_calls", [])
            if not tool_calls:
                break  # No more tool calls, we're done
            
            new_tool_activity = False
            audio_generated = False
            
            for tool_call in tool_calls:
                try:
                    function_name = tool_call.get("function", {}).get("name")
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    
                    if function_name == "generate_audio":
                        result = self._execute_generate_audio(arguments)
                        results.append({"function_name": function_name, "result": result})
                        if result.get("success"):
                            audio_path = result.get("audio_path")
                            # Once audio is generated, stop further tool execution and exit loop
                            audio_generated = True
                            new_tool_activity = True
                            break
                    elif function_name == "visual_analysis":
                        result = self._execute_visual_analysis(arguments)
                        results.append({"function_name": function_name, "result": result})
                        if result.get("success"):
                            # Inject analysis back into the conversation as assistant context
                            visual_analysis_content = result.get('analysis', '')
                            current_messages.append({
                                "role": "assistant",
                                "content": f"Visual analysis result: {visual_analysis_content}"
                            })
                            new_tool_activity = True
                    else:
                        result = {"success": False, "error": f"Unknown function: {function_name}"}
                        results.append({"function_name": function_name or "unknown", "result": result})
                except Exception as e:
                    results.append({
                        "function_name": tool_call.get("function", {}).get("name", "unknown"),
                        "result": {"success": False, "error": str(e)}
                    })
            
            # If audio was generated, stop the agent loop immediately to return response to user
            if audio_generated:
                break

            # If tools ran and produced new context (e.g., visual analysis), ask LLM again
            if new_tool_activity:
                if self.logger:
                    self.logger.log_result("🤖 Agent: Follow-up LLM Call ", {
                        "action": "follow_up_llm_call",
                        "content": "Continuing with LLM after tool execution",
                        "iteration": iteration
                    })
                current_response = self._get_llm_response(current_messages)
                # Update stored response text
                if current_response.get("content"):
                    self.last_response_text = current_response.get("content", "")
                continue
            else:
                break
        
        return {
            "success": True,
            "audio_path": audio_path,
            "transcription": session_context.get("last_transcription", ""),
            "response_text": getattr(self, 'last_response_text', current_response.get("content", "")),
            "tool_results": results
        }
    
    def _execute_generate_audio(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio generation."""
        try:
            text = arguments.get("text", "")
            voice = arguments.get("voice", "belinda")
            
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
            # If a logger is present, persist audio artifact in the run directory
            if hasattr(self, 'logger') and self.logger:
                try:
                    with open(output_path, 'rb') as f:
                        audio_bytes = f.read()
                    saved_rel = f"audio_{int(time.time()*1000)}.wav"
                    saved_path = self.logger.save_file(saved_rel, audio_bytes, mode='binary')
                    # Prefer returning the saved artifact path
                    output_path = saved_path
                except Exception as e:
                    if self.logger:
                        self.logger.log_warning("audio_artifact_save_failed", str(e))
            
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
                # Persist analysis artifact as text file in run directory
                try:
                    saved_rel = f"visual_analysis_{int(time.time()*1000)}.txt"
                    self.logger.save_file(saved_rel, analysis, mode='text')
                except Exception as e:
                    self.logger.log_warning("visual_analysis_artifact_save_failed", str(e))
                
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
                voice="belindaan",
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
    
