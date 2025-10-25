'''
Boson Audio understanding support.
'''

import os
import base64
import openai

class AudioUnderstanding:

    def __init__(self, model_name: str = "higgs-audio-understanding-Hackathon", api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        self.client = openai.Client(api_key=api_key, base_url="https://hackathon.boson.ai/v1")
        self.name = model_name
    
    def _encode_audio_to_base64(self, file_path: str) -> str:
        """Encode audio file to base64 format."""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    
    def _is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        return path.startswith('http://') or path.startswith('https://')
    
    def call(self, 
        audio: str,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str | None = None,
        verbose: bool = False
        ) -> str:
        """
        Call the audio understanding model.
        
        Args:
            audio: Path to audio file or URL
            system_prompt: System instruction (default: "You are a helpful assistant.")
            user_prompt: Optional additional text prompt for chat/questions about the audio
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
        
        response = self.client.chat.completions.create(
            model=self.name,
            messages=assembled_payloads,
            max_completion_tokens=256,
            temperature=0.0
        )
        
        if verbose:
            print(f"[DEBUG] Model: {self.name}")
            print(f"[DEBUG] Audio: {audio}")
            print(f"[DEBUG] Response: {response.choices[0].message.content}")
        
        return response.choices[0].message.content


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