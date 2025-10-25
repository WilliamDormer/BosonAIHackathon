"""
Voice Sight Session Management.
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime


class VoiceSightSession:
    """Manages the state of a Voice Sight conversation session."""
    
    def __init__(self):
        """Initialize a new session."""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        self.state = "initialized"
        self.conversation_history = []
        self.current_context = {}
        self.language_preferences = {}
        self.translation_mode = False
        
    def add_message(self, role: str, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
    
    def get_conversation_context(self, max_messages: int = 10) -> list:
        """Get recent conversation context."""
        return self.conversation_history[-max_messages:] if self.conversation_history else []
    
    def set_language_preferences(self, from_lang: str, to_lang: str):
        """Set language preferences for translation."""
        self.language_preferences = {
            "from_language": from_lang,
            "to_language": to_lang
        }
        self.translation_mode = True
        self.state = "translation_ready"
    
    def get_language_preferences(self) -> Optional[Dict[str, str]]:
        """Get current language preferences."""
        return self.language_preferences if self.language_preferences else None
    
    def set_state(self, state: str):
        """Set the current session state."""
        self.state = state
    
    def get_state(self) -> str:
        """Get the current session state."""
        return self.state
    
    def reset(self):
        """Reset the session to initial state."""
        self.state = "initialized"
        self.conversation_history = []
        self.current_context = {}
        self.language_preferences = {}
        self.translation_mode = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "state": self.state,
            "conversation_history": self.conversation_history,
            "current_context": self.current_context,
            "language_preferences": self.language_preferences,
            "translation_mode": self.translation_mode
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save session to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VoiceSightSession':
        """Load session from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = cls()
        session.session_id = data.get("session_id", session.session_id)
        session.state = data.get("state", "initialized")
        session.conversation_history = data.get("conversation_history", [])
        session.current_context = data.get("current_context", {})
        session.language_preferences = data.get("language_preferences", {})
        session.translation_mode = data.get("translation_mode", False)
        
        return session
