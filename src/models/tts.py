'''
Boson text-to-speech support.
'''

import os
import base64
import wave
import openai
from typing import Iterator
from collections.abc import Iterator as ABCIterator

class TTS:

    def __init__(self, model_name: str = "higgs-audio-generation-Hackathon", api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        self.client = openai.Client(api_key=api_key, base_url="https://hackathon.boson.ai/v1")
        self.name = model_name
    
    def _encode_audio_to_base64(self, file_path: str) -> str:
        """Encode audio file to base64 format."""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    
    def _pcm_to_wav(self, pcm_data: bytes, output_path: str, 
                    num_channels: int = 1, sample_width: int = 2, sample_rate: int = 24000):
        """Convert PCM data to WAV file."""
        with wave.open(output_path, 'wb') as wav:
            wav.setnchannels(num_channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
    
    def generate_simple(self, 
        text: str,
        voice: str = "en_woman",
        output_path: str = "output.wav",
        verbose: bool = False
        ) -> bytes:
        """
        Simple text-to-speech generation with predefined voices.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use. Options: belinda, broom_salesman, chadwick, en_man, 
                   en_woman, mabel, vex, zh_man_sichuan
            output_path: Path to save the output WAV file
            verbose: Enable debug output
        
        Returns:
            PCM audio data as bytes
        """
        
        if verbose:
            print(f"[DEBUG] Generating speech with voice: {voice}")
            print(f"[DEBUG] Text: {text}")
        
        # Generate audio using simple API
        response = self.client.audio.speech.create(
            model=self.name,
            voice=voice,
            input=text,
            response_format="pcm"
        )
        
        pcm_data = response.content
        
        # Save to WAV file
        self._pcm_to_wav(pcm_data, output_path)
        
        if verbose:
            print(f"[DEBUG] Audio saved to: {output_path}")
        
        return pcm_data
    
    def generate_with_reference(self,
        text: str,
        reference_audio: str,
        reference_transcript: str,
        output_path: str = "output.wav",
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_completion_tokens: int = 4096,
        verbose: bool = False
        ) -> bytes | Iterator[bytes]:
        """
        Generate speech with voice cloning using reference audio.
        
        Args:
            text: Text to convert to speech
            reference_audio: Path to reference audio file for voice cloning
            reference_transcript: Transcript of the reference audio
            output_path: Path to save the output WAV file (only for non-streaming)
            stream: Enable streaming mode
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 50)
            max_completion_tokens: Maximum tokens to generate (default: 4096)
            verbose: Enable debug output
        
        Returns:
            Audio data as bytes (static mode) or Iterator of bytes (streaming mode)
        """
        
        if verbose:
            print("[DEBUG] Generating speech with voice cloning")
            print(f"[DEBUG] Reference audio: {reference_audio}")
            print(f"[DEBUG] Text: {text}")
            print(f"[DEBUG] Streaming: {stream}")
        
        # Encode reference audio
        reference_b64 = self._encode_audio_to_base64(reference_audio)
        reference_format = reference_audio.split(".")[-1]
        
        # Build messages for voice cloning
        messages = [
            {"role": "user", "content": reference_transcript},
            {
                "role": "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": reference_b64, "format": reference_format}
                }],
            },
            {"role": "user", "content": text},
        ]
        
        # Generate audio with reference
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            modalities=["text", "audio"],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": top_k},
        )
        
        if stream:
            # Streaming mode
            if verbose:
                print("[DEBUG] Streaming audio...")
            
            def audio_stream_generator() -> Iterator[bytes]:
                for chunk in response:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'audio'):
                        audio_data = chunk.choices[0].delta.audio
                        if audio_data:
                            # Handle both dict and object responses
                            if isinstance(audio_data, dict):
                                audio_b64 = audio_data.get('data')
                            else:
                                audio_b64 = audio_data.data if hasattr(audio_data, 'data') else None
                            
                            if audio_b64:
                                audio_chunk: bytes = base64.b64decode(audio_b64)
                                yield audio_chunk
            
            return audio_stream_generator()
        else:
            # Static mode
            audio_b64 = response.choices[0].message.audio.data
            audio_data = base64.b64decode(audio_b64)
            
            # Save to file
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            if verbose:
                print(f"[DEBUG] Audio saved to: {output_path}")
            
            return audio_data


if __name__ == "__main__":
    
    # debug mode - test the TTS model
    
    api_key = "bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
    
    tts = TTS(api_key_override=api_key)
    
    # Example 1: Simple generation with predefined voice
    print("=== Example 1: Simple TTS ===")
    tts.generate_simple(
        text="Hello, this is a test of the audio generation system.",
        voice="belinda",
        output_path="./tests/test_tts_output/test_simple.wav",
        verbose=True
    )
    print("Simple TTS completed!\n")
    
    # Example 2: Voice cloning with reference audio (static)
    print("=== Example 2: Voice Cloning (Static) ===")
    tts.generate_with_reference(
        text="Welcome to Boson AI's voice generation system.",
        reference_audio="../../ref-audio/mabel.wav",
        reference_transcript="This is a reference audio sample for voice cloning.",
        output_path="./tests/test_tts_output/test_cloned.wav",
        stream=False,
        verbose=True
    )
    print("Voice cloning completed!\n")
    
    # Example 3: Voice cloning with streaming
    print("=== Example 3: Voice Cloning (Streaming) ===")
    audio_stream = tts.generate_with_reference(
        text="This is a streaming test.",
        reference_audio="../../ref-audio/hogwarts_wand_seller_v2.wav",
        reference_transcript="This is a reference audio for streaming.",
        stream=True,
        verbose=True
    )
    
    # Collect streamed audio chunks
    audio_chunks: list[bytes] = []
    if isinstance(audio_stream, ABCIterator):
        for chunk in audio_stream:
            audio_chunks.append(chunk)
            print(f"[DEBUG] Received chunk of {len(chunk)} bytes")
    
    # Save streamed audio with proper WAV headers
    if audio_chunks:
        # Concatenate all chunks
        pcm_data = b"".join(audio_chunks)
        # Convert to WAV using the helper method (assumes 24kHz, mono, 16-bit)
        tts._pcm_to_wav(pcm_data, "./tests/test_tts_output/test_streamed.wav")
        print("Streaming TTS completed!")