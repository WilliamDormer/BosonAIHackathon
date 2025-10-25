'''
Large Language Model support.
'''

import os
import openai

class LLM:

    def __init__(self, use_thinking: bool = False, api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        self.client = openai.Client(api_key=api_key, base_url="https://hackathon.boson.ai/v1")
        self.use_thinking = use_thinking
        self.name = "Qwen3-32B-thinking-Hackathon" if use_thinking else "Qwen3-32B-non-thinking-Hackathon"
    
    def call(self, 
        system_prompt: str = "You are a helpful assistant.", 
        user_prompt: str = "",
        verbose: bool = False
        ) -> str:
        
        # Build the messages
        assembled_payloads = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add user message if provided
        if user_prompt:
            assembled_payloads.append({
                "role": "user",
                "content": user_prompt
            })
        
        response = self.client.chat.completions.create(
            model=self.name,
            messages=assembled_payloads,
            max_tokens=256,
            temperature=0.2
        )
        
        if verbose:
            print(f"[DEBUG] Model: {self.name}")
            print(f"[DEBUG] Response: {response.choices[0].message.content}")
        
        return response.choices[0].message.content
    

if __name__ == "__main__":
    
    # debug mode - test the LLM with both thinking and non-thinking models
    
    api_key = "bai-CeNIcRcYQqe50mhRO9vlnJvdImMRXfBQIkeMKovGGR9fa4Ke"
    
    # Example 1: Non-thinking model
    llm_non_thinking = LLM(use_thinking=False, api_key_override=api_key)
    response1 = llm_non_thinking.call(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        verbose=True
    )
    print(f"Non-thinking Response: {response1}")
    print()
    
    # Example 2: Thinking model
    llm_thinking = LLM(use_thinking=True, api_key_override=api_key)
    response2 = llm_thinking.call(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        verbose=True
    )
    print(f"Thinking Response: {response2}")

