'''
Backup ASR using Qwen3-Omni for audio speech recognition.
Based on Qwen3-Omni-30B-A3B-Thinking-Hackathon multimodal capabilities.
'''

import os
import json
import time
import yaml  # type: ignore
import base64
import openai
from typing import Optional, Dict, Any

# MODEL OUTPUT BROKEN! DO NOT USE FOR NOW

class ASR:
    """
    Backup ASR implementation using Qwen3-Omni multimodal model.
    Supports audio speech recognition via the Qwen3-Omni-30B-A3B-Thinking model.
    """

    def __init__(self, model_name: str = "Qwen3-Omni-30B-A3B-Thinking-Hackathon", api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        # Load config
        self.config = self._load_config()
        base_url = self.config.get('api', {}).get('base_url', 'https://hackathon.boson.ai/v1')
        
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        
        # Model configuration - using Qwen3-Omni
        self.name = model_name
        
        # Get ASR config or use sensible defaults for Qwen3-Omni
        asr_config = self.config.get('asr', {})
        self.default_max_tokens = asr_config.get('max_completion_tokens', 512)
        self.default_temperature = asr_config.get('temperature', 0.2)  # Lower for transcription
        self.default_top_p = asr_config.get('top_p', 1.0)
        
        # Retry configuration
        retry_config = self.config.get('retry', {})
        self.max_retries = retry_config.get('max_retries', 5)
        self.initial_delay = retry_config.get('initial_delay', 0.5)
        self.backoff_multiplier = retry_config.get('backoff_multiplier', 1.5)
        self.max_delay = retry_config.get('max_delay', 5.0)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}. Using defaults.")
            return {}
    
    def _encode_audio_to_base64(self, file_path: str) -> str:
        """Encode audio file to base64 format."""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    
    def _is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        return path.startswith('http://') or path.startswith('https://')
    
    def _build_audio_url(self, audio_path: str) -> str:
        """
        Build audio URL for Qwen3-Omni.
        If it's already a URL, return as-is.
        If it's a local file, encode to base64 data URL.
        """
        if self._is_url(audio_path):
            return audio_path
        else:
            # For local files, we'll use base64 encoding
            # Format: data:audio/wav;base64,<base64_data>
            audio_base64 = self._encode_audio_to_base64(audio_path)
            file_ext = audio_path.split(".")[-1].lower()
            mime_type = f"audio/{file_ext}" if file_ext in ['wav', 'mp3', 'ogg', 'flac'] else "audio/wav"
            return f"data:{mime_type};base64,{audio_base64}"
    
    def _validate_json_response(self, response: str) -> bool:
        """
        Validate if response contains valid JSON dictionary.
        
        Args:
            response: Response string to validate
        
        Returns:
            True if valid JSON dict found, False otherwise
        """
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end].strip()
                parsed = json.loads(json_str)  # Validate JSON parsing
                # Must be a dictionary
                return isinstance(parsed, dict) and len(parsed) > 0
            return False
        except (json.JSONDecodeError, Exception):
            return False
    
    def call(self, 
        audio: str,
        system_prompt="You are an expert transcription assistant.",
        user_prompt='Transcribe this audio accurately, output only the text without any additional commentary. Place your output in the json: {"response": "the transcription of the audio"}',
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        verbose: bool = False
        ) -> str:
        """
        Call Qwen3-Omni for audio understanding with retry logic.
        
        Args:
            audio: Path to audio file or URL
            system_prompt: System instruction (default: "You are a helpful assistant.")
            user_prompt: Optional additional text prompt for chat/questions about the audio
            max_tokens: Maximum tokens to generate (uses config default if not specified)
            temperature: Sampling temperature (uses config default if not specified)
            validate_json: If True, validates response is valid JSON and retries if not
            verbose: Enable debug output
        
        Returns:
            Model response as string
        """
        
        # manually override
        system_prompt="You are an expert transcription assistant.",
        user_prompt='Transcribe this audio accurately, output only the text without any additional commentary. Place your output in the json: {"response": "the transcription of the audio"}',
        
        # Build the messages using Qwen3-Omni multimodal format
        assembled_payloads = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Build audio URL (supports both local files and URLs)
        audio_url = self._build_audio_url(audio)
        
        # Build user message with audio using Qwen3-Omni's audio_url format
        # According to API doc: {"type": "audio_url", "audio_url": {"url": "..."}}
        user_content_items = [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            }
        ]
        
        # Add text instruction if provided, or use default transcription instruction
        text_instruction = user_prompt if user_prompt else "Transcribe this audio accurately."
        user_content_items.append({
            "type": "text",
            "text": text_instruction
        })
        
        assembled_payloads.append({
            "role": "user",
            "content": user_content_items  # type: ignore
        })
        
        # Use config defaults if not specified
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Retry loop
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=assembled_payloads,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                print(response)
                
                content = response.choices[0].message.content
                
                # Handle None response
                if content is None:
                    raise ValueError("Received None response from model")
                
                # # Validate JSON if required
                # if validate_json:
                #     if not self._validate_json_response(content):
                #         if verbose:
                #             print(f"[DEBUG] Attempt {attempt + 1}: Invalid JSON response, retrying...")
                #             print(f"[DEBUG] Full response: {content}")
                #         raise ValueError(f"Response is not valid JSON (length={len(content)})")

                # get the last json body: "{"response": "<content>"}" of the string and get the json body only
                try:
                    import re
                    response_match = re.search(r'{"response"\s*:\s*"(.*?)"}', content)
                    if response_match:
                        content = response_match.group(0)
                except Exception as e:
                    if verbose:
                        print(f"[DEBUG] Failed to extract JSON response: {e}")
                            
                if verbose:
                    print(f"[DEBUG] Model: {self.name} (Qwen3-Omni)")
                    print(f"[DEBUG] Audio: {audio}")
                    print(f"[DEBUG] Attempt: {attempt + 1}")
                    print(f"[DEBUG] Response: {content}")
                
                return content
                
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    if verbose:
                        print(f"[DEBUG] Attempt {attempt + 1} failed: {str(e)}")
                        print(f"[DEBUG] Retrying in {delay:.2f} seconds...")
                    
                    time.sleep(delay)
                    delay = min(delay * self.backoff_multiplier, self.max_delay)
                else:
                    if verbose:
                        print(f"[DEBUG] All {self.max_retries} attempts failed")
        
        # If all retries failed, raise the last error
        raise Exception(f"ASR (Qwen3-Omni) call failed after {self.max_retries} attempts: {str(last_error)}")


if __name__ == "__main__":
    
    # Debug mode - test the Qwen3-Omni audio understanding
    
    api_key = "bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
    
    if not api_key:
        print("Error: BOSON_API_KEY environment variable not set")
        exit(1)
    
    audio_model = ASR(api_key_override=api_key)
    
    # Example 1: Audio Transcription
    print("=== Example 1: Audio Transcription (Qwen3-Omni) ===")
    try:
        response1 = audio_model.call(
            audio="/Users/tomlu/Documents/GitHub/BosonAIHackathon/ref-audio/belinda.wav",
            system_prompt="You are an expert transcription assistant.",
            user_prompt='Transcribe this audio accurately, output only the text without any additional commentary. Place your output in the json: {"response": "the transcription of the audio"}',
            verbose=True
        )
        print(f"Transcription: {response1}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # # Example 2: Audio chat - ask questions about the audio
    # print("=== Example 2: Audio Analysis (Qwen3-Omni) ===")
    # try:
    #     response2 = audio_model.call(
    #         audio="../../ref-audio/en_woman.wav",
    #         system_prompt="You are a helpful assistant.",
    #         user_prompt="Analyze this audio: Is it a male's voice or female's? What is the tone and emotion?",
    #         verbose=True
    #     )
    #     print(f"Analysis Response: {response2}")
    # except Exception as e:
    #     print(f"Error: {e}")
    # print()
    
    # # Example 3: General audio understanding
    # print("=== Example 3: Detailed Audio Understanding (Qwen3-Omni) ===")
    # try:
    #     response3 = audio_model.call(
    #         audio="../../ref-audio/en_man.wav",
    #         system_prompt="You are an audio analysis expert.",
    #         user_prompt="Describe this audio in detail: speaker characteristics, content, tone, and any background sounds.",
    #         verbose=True
    #     )
    #     print(f"Understanding: {response3}")
    # except Exception as e:
    #     print(f"Error: {e}")
