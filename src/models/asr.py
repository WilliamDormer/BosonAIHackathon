'''
Boson Audio understanding support.
'''

import os
import json
import time
import yaml
import base64
import openai
from typing import Optional, Dict, Any

class AudioUnderstanding:

    def __init__(self, model_name: str = "higgs-audio-understanding-Hackathon", api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        # Load config
        self.config = self._load_config()
        base_url = self.config.get('api', {}).get('base_url', 'https://hackathon.boson.ai/v1')
        
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        
        # Get model configuration
        asr_config = self.config.get('asr', {})
        self.name = asr_config.get('model', model_name)
        self.default_max_tokens = asr_config.get('max_completion_tokens', 512)
        self.default_temperature = asr_config.get('temperature', 0.0)
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
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str | None = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validate_json: bool = False,
        verbose: bool = False
        ) -> str:
        """
        Call the audio understanding model with retry logic.
        
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
        
        # Build the messages
        assembled_payloads = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Encode audio
        audio_data = audio if self._is_url(audio) else self._encode_audio_to_base64(audio)
        file_format = audio.split(".")[-1] if not self._is_url(audio) else "wav"
        
        # Build user message with audio
        user_content = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_data,
                    "format": file_format,
                },
            }
        ]
        
        assembled_payloads.append({
            "role": "user",
            "content": user_content  # type: ignore
        })
        
        # Add additional text prompt if provided (for chat/questions about audio)
        if user_prompt:
            assembled_payloads.append({
                "role": "user",
                "content": user_prompt
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
                    messages=assembled_payloads,
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                
                # Handle None response
                if content is None:
                    raise ValueError("Received None response from model")
                
                # Validate JSON if required
                if validate_json:
                    if not self._validate_json_response(content):
                        if verbose:
                            print(f"[DEBUG] Attempt {attempt + 1}: Invalid JSON response, retrying...")
                            print(f"[DEBUG] Full response: {content}")
                        raise ValueError(f"Response is not valid JSON (length={len(content)})")
                
                if verbose:
                    print(f"[DEBUG] Model: {self.name}")
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
        raise Exception(f"ASR call failed after {self.max_retries} attempts: {str(last_error)}")


if __name__ == "__main__":
    
    # debug mode - test the audio understanding model
    
    api_key = "bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
    
    audio_model = AudioUnderstanding(api_key_override=api_key)
    
    # Example 1: Transcription
    print("=== Example 1: Audio Transcription ===")
    response1 = audio_model.call(
        audio="../../ref-audio/en_woman.wav",
        system_prompt="Transcribe this audio for me.",
        verbose=True
    )
    print(f"Transcription: {response1}")
    print()
    
    # Example 2: Audio chat - ask questions about the audio
    print("=== Example 2: Audio Chat ===")
    response2 = audio_model.call(
        audio="../../ref-audio/en_woman.wav",
        system_prompt="You are a helpful assistant.",
        user_prompt="Is it a male's voice or female's?",
        verbose=True
    )
    print(f"Chat Response: {response2}")
    print()
    
    # Example 3: General audio understanding
    print("=== Example 3: General Understanding ===")
    response3 = audio_model.call(
        audio="../../ref-audio/en_man.wav",
        system_prompt="Describe the audio in detail, including the speaker's characteristics and content.",
        verbose=True
    )
    print(f"Understanding: {response3}")