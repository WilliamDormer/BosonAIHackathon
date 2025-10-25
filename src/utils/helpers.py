"""
Utility functions for JSON parsing and audio processing.
"""

import json
import re
from typing import Dict, Any, Optional, Union


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response, handling cases where extra text is present.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not response_text:
        return None
    
    # Try to find JSON object in the response
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array
    array_pattern = r'\[.*\]'
    array_match = re.search(array_pattern, response_text, re.DOTALL)
    
    if array_match:
        try:
            json_str = array_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, try parsing the entire response
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text: str, key: str = None) -> Optional[Union[Dict[str, Any], Any]]:
    """
    Extract JSON value from text, optionally by key.
    
    Args:
        text: Text containing JSON
        key: Optional key to extract from JSON
        
    Returns:
        Extracted JSON value or None if not found
    """
    json_data = parse_json_response(text)
    if json_data is None:
        return None
    
    if key is None:
        return json_data
    
    return json_data.get(key)


def validate_json_structure(data: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that JSON data contains required keys.
    
    Args:
        data: JSON data to validate
        required_keys: List of required keys
        
    Returns:
        True if all required keys are present, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    return all(key in data for key in required_keys)


def format_json_for_llm(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format JSON data for LLM consumption.
    
    Args:
        data: JSON data to format
        indent: JSON indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def create_tool_call_schema(tool_name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a tool call schema for LLM function calling.
    
    Args:
        tool_name: Name of the tool
        description: Description of the tool
        parameters: JSON schema for parameters
        
    Returns:
        Tool call schema dictionary
    """
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters
        }
    }


def encode_audio_to_base64(file_path: str) -> str:
    """
    Encode audio file to base64 format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Base64 encoded audio data
    """
    import base64
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def get_audio_format(file_path: str) -> str:
    """
    Get audio format from file extension.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Audio format (e.g., 'wav', 'mp3')
    """
    return file_path.split(".")[-1].lower()


def create_audio_message(audio_path: str, instruction: str = None) -> Dict[str, Any]:
    """
    Create a message with audio input for multimodal LLM.
    
    Args:
        audio_path: Path to audio file
        instruction: Optional text instruction
        
    Returns:
        Message dictionary with audio content
    """
    audio_base64 = encode_audio_to_base64(audio_path)
    audio_format = get_audio_format(audio_path)
    
    content = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_base64,
                "format": audio_format,
            },
        },
    ]
    
    if instruction:
        content.append({
            "type": "text",
            "text": instruction
        })
    
    return {
        "role": "user",
        "content": content
    }


def create_text_message(text: str) -> Dict[str, Any]:
    """
    Create a simple text message.
    
    Args:
        text: Text content
        
    Returns:
        Message dictionary
    """
    return {
        "role": "user",
        "content": text
    }


def create_assistant_message(content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create an assistant message.
    
    Args:
        content: Message content (text or audio)
        
    Returns:
        Message dictionary
    """
    return {
        "role": "assistant",
        "content": content
    }