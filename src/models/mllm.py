'''
Multimodal Large Language Model support.
'''

import os
import base64
import openai

class MLLM:

    def __init__(self, model_name: str = "Qwen3-Omni-30B-A3B-Thinking-Hackathon", api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        self.client = openai.Client(api_key=api_key, base_url="https://hackathon.boson.ai/v1")
        self.name = model_name
    
    def _encode_file_to_base64(self, file_path: str) -> str:
        """Encode a local file to base64 data URI."""
        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine the file type based on extension
        ext = file_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif ext == 'png':
            mime_type = 'image/png'
        else:
            mime_type = 'application/octet-stream'
        
        return f"data:{mime_type};base64,{file_data}"
    
    def _is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        return path.startswith('http://') or path.startswith('https://')
        
    def call(self, 
        system_prompt: str = "You are a helpful assistant.", 
        user_prompt: str = "",
        image: str | None = None, 
        audio: str | None = None, 
        verbose: bool = False
        ) -> str:
        
        # Build the system message
        assembled_payloads = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Build user message content array with multimodal inputs
        user_content = []
        
        # Add image if provided
        if image:
            # Convert local file to base64 data URI if it's not a URL
            image_url = image if self._is_url(image) else self._encode_file_to_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        # Add audio if provided
        if audio:
            # Convert local file to base64 data URI if it's not a URL
            audio_url = audio if self._is_url(audio) else self._encode_file_to_base64(audio)
            user_content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_url,
                    "format": "wav"
                }
            })
        
        # Add user message with all content
        if user_content:
            assembled_payloads.append({
                "role": "user",
                "content": user_content  # type: ignore
            })
        
        
        response = self.client.chat.completions.create(
            model=self.name,
            messages=assembled_payloads,
            max_tokens=256,
            temperature=0.2
        )
        
        if verbose:
            print(f"[DEBUG] Response: {response.choices[0].message.content}")
            
        
        return response.choices[0].message.content
    
    
if __name__ == "__main__":
    
    # debug mode - test the MLLM with multimodal inputs
    
    api_key = "bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
    
    mllm = MLLM("Qwen3-Omni-30B-A3B-Thinking-Hackathon", api_key_override=api_key)
    
    # Example 1: Text + Image + Audio from URLs
    response = mllm.call(
        system_prompt="You are a helpful assistant.",
        user_prompt="What do you see in this image?",
        image="./tests/test_mllm_input/cars.jpg",
        verbose=True
    )
    print(f"Response 1: {response}")
    
    