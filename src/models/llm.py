'''
Large Language Model support.
'''

import os
import json
import time
import yaml  # type: ignore
import openai
from typing import Optional, Dict, Any, List, Union

class LLM:

    def __init__(self, use_thinking: bool = False, api_key_override: str | None = None):
        
        api_key = api_key_override if api_key_override else os.getenv("BOSON_API_KEY")
        
        # Load config
        self.config = self._load_config()
        base_url = self.config.get('api', {}).get('base_url', 'https://hackathon.boson.ai/v1')
        
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        self.use_thinking = use_thinking
        
        # Get model configuration
        config_key = 'thinking' if use_thinking else 'non_thinking'
        model_config = self.config.get('llm', {}).get(config_key, {})
        
        self.name = model_config.get('model', 
            "Qwen3-32B-thinking-Hackathon" if use_thinking else "Qwen3-32B-non-thinking-Hackathon")
        self.default_max_tokens = model_config.get('max_tokens', 512)
        self.default_temperature = model_config.get('temperature', 0.2)
        self.default_top_p = model_config.get('top_p', 0.95)
        
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
        system_prompt: str = "You are a helpful assistant.", 
        user_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validate_json: bool = False,
        verbose: bool = False
        ) -> str:
        """
        Call the LLM with retry logic and optional JSON validation.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
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
        
        # Add user message if provided
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
                    messages=assembled_payloads,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.default_top_p
                )
                
                content = response.choices[0].message.content or ""  # Handle None content
                
                # Validate JSON if required
                if validate_json:
                    if not self._validate_json_response(content):
                        if verbose:
                            print(f"[DEBUG] Attempt {attempt + 1}: Invalid JSON response, retrying...")
                            print(f"[DEBUG] Full response: {content}")
                        raise ValueError(f"Response is not valid JSON (length={len(content)})")
                
                if verbose:
                    print(f"[DEBUG] Model: {self.name}")
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
        raise Exception(f"LLM call failed after {self.max_retries} attempts: {str(last_error)}")
    
    def chat_completion(
        self, 
        assembled_payloads: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validate_json: bool = False,
        verbose: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
        ) -> Union[str, Any]:  # Returns str without tools, message object with tools
        """
        Call the LLM with retry logic and optional JSON validation.
        
        Args:
            assembled_payloads: List of messages that has already been assembled
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            validate_json: If True, validates response is valid JSON
            verbose: Enable debug output
            tools: List of available tools for function calling (OpenAI format)
            tool_choice: Control tool calling behavior ("auto", "none", or {"type": "function", "function": {"name": "my_function"}})
        
        Returns:
            Either the message content string (if no tools) or the full response object (if tools used)
        """
        
        # Use config defaults if not specified
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Retry loop
        delay = self.initial_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Build API call parameters
                api_params = {
                    "model": self.name,
                    "messages": assembled_payloads,  # type: ignore
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": self.default_top_p
                }
                
                # Add tools if provided
                if tools:
                    api_params["tools"] = tools
                    if tool_choice:
                        api_params["tool_choice"] = tool_choice
                
                response = self.client.chat.completions.create(**api_params)
                
                # Get the message from the response
                message = response.choices[0].message
                content = message.content or ""  # Handle None content
                
                # If tools were provided, return the full message object (includes tool_calls)
                if tools:
                    if verbose:
                        print(f"[DEBUG] Model: {self.name}")
                        print(f"[DEBUG] Attempt: {attempt + 1}")
                        print(f"[DEBUG] Content: {content}")
                        print(f"[DEBUG] Tool calls: {message.tool_calls}")
                    return message  # Return full message object with tool_calls
                
                # Otherwise validate and return content as string
                # Validate JSON if required
                if validate_json:
                    if not self._validate_json_response(content):
                        if verbose:
                            print(f"[DEBUG] Attempt {attempt + 1}: Invalid JSON response, retrying...")
                            print(f"[DEBUG] Full response: {content}")
                        raise ValueError(f"Response is not valid JSON (length={len(content)})")
                
                if verbose:
                    print(f"[DEBUG] Model: {self.name}")
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
        raise Exception(f"LLM call failed after {self.max_retries} attempts: {str(last_error)}")
    
        

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

