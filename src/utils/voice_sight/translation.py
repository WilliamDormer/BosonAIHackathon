"""
Translation utilities for audio-based translation pipeline.
"""

import os
import sys
import tempfile
import json
import yaml  # type: ignore
from typing import Optional, Tuple, Dict, Any

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.tts import TTS  # type: ignore
from models.asr import AudioUnderstanding  # type: ignore
from models.llm import LLM  # type: ignore


class TranslationPipeline:
    """Audio-based translation pipeline supporting Chinese-English translation."""
    
    def __init__(self, api_key: Optional[str] = None, logger: Optional[Any] = None):
        """Initialize translation pipeline with necessary models."""
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.logger = logger
        
        # Initialize models
        self.tts = TTS(api_key_override=self.api_key)
        self.asr = AudioUnderstanding(api_key_override=self.api_key)
        self.llm = LLM(use_thinking=False, api_key_override=self.api_key)
        
        # Load prompts from YAML
        self.prompts = self._load_prompts()
        
        # Extract mappings from prompts
        self.language_names = self.prompts['translation']['language_names']
        self.voice_map = self.prompts['translation']['voice_mappings']
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration file."""
        prompts_path = os.path.join(parent_dir, 'prompts', 'voice_sight.yaml')
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            if self.logger:
                self.logger.log_error("prompt_loading", f"Failed to load prompts: {str(e)}")
            # Fallback to basic prompts
            return {
                'translation': {
                    'language_names': {'en': 'English', 'zh': 'Chinese'},
                    'voice_mappings': {'en': 'en_woman', 'zh': 'zh_man_sichuan'},
                    'greetings': {
                        'initial': 'Hello! Welcome to the translation service.',
                        'confirmation': 'Please confirm.',
                    }
                }
            }
    
    def _parse_json_response(self, response: str, step: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from LLM, with error handling.
        
        Args:
            response: Raw response string from LLM
            step: Name of the step (for logging)
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            # Try to find JSON in the response
            # Sometimes LLM might add extra text, so we look for {...}
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                # Clean up any escaped newlines or extra whitespace in the JSON string
                json_str = json_str.strip()
                parsed = json.loads(json_str)
                
                # Validate it's actually a dictionary
                if not isinstance(parsed, dict):
                    if self.logger:
                        # Log full response on error
                        self.logger.log_error(f"{step}_parse", f"Parsed JSON is not a dict: {type(parsed)}")
                        self.logger.save_file(f"{step}_raw_response_not_dict.txt", response, mode='text')
                    return None
                
                if self.logger:
                    self.logger.log_step(f"{step}_parsed", parsed)
                
                return parsed
            else:
                if self.logger:
                    # Save full response when no JSON found
                    self.logger.log_error(f"{step}_parse", f"No JSON found in response (length: {len(response)})")
                    self.logger.save_file(f"{step}_raw_response_no_json.txt", response, mode='text')
                return None
                
        except (json.JSONDecodeError, Exception) as e:
            if self.logger:
                # Save full response on JSON decode errors
                error_msg = f"JSON parse error: {type(e).__name__}: {str(e)}"
                self.logger.log_error(f"{step}_parse", error_msg)
                self.logger.save_file(f"{step}_raw_response_error.txt", 
                    f"Error: {error_msg}\n\n{'='*80}\nFull Response:\n{'='*80}\n\n{response}", 
                    mode='text')
            return None
    
    def generate_audio_prompt(self, text: str, language: str = "en", output_path: Optional[str] = None) -> str:
        """
        Generate audio prompt using TTS.
        
        Args:
            text: Text to convert to speech
            language: Language code ('en' or 'zh')
            output_path: Optional path to save audio file
        
        Returns:
            Path to generated audio file
        """
        if self.logger:
            self.logger.log_step("tts_request", {
                "text": text,
                "language": language,
                "voice": self.voice_map.get(language, "en_woman")
            })
        
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        
        voice = self.voice_map.get(language, "en_woman")
        self.tts.generate_simple(text, voice=voice, output_path=output_path)
        
        if self.logger:
            self.logger.log_step("tts_complete", {"output_path": output_path})
        
        return output_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using JSON response format with retry logic.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        if self.logger:
            self.logger.log_step("asr_request", {"audio_path": audio_path})
        
        system_prompt = self.prompts['translation']['transcription']['system']
        
        # Call with JSON validation enabled - will retry automatically if JSON is invalid
        response = self.asr.call(
            audio=audio_path, 
            system_prompt=system_prompt,
            validate_json=True,
            verbose=False
        )
        
        # Save raw response for debugging
        if self.logger:
            self.logger.save_file(f"transcription_raw_response_{self.logger.step_counter}.txt", response, mode='text')
        
        # Parse JSON response (should always be valid after retries)
        parsed = self._parse_json_response(response, "transcription")
        
        if parsed and 'transcribed_text' in parsed:
            transcribed_text = parsed['transcribed_text']
            
            if self.logger:
                self.logger.log_step("asr_complete", {
                    "transcribed_text": transcribed_text,
                    "confidence": parsed.get('confidence', 'unknown'),
                    "language_detected": parsed.get('language_detected', 'unknown')
                })
            
            return transcribed_text
        else:
            # This should rarely happen now due to retry logic
            if self.logger:
                self.logger.log_warning("asr_fallback", "Using raw response as transcription")
            return response
    
    def parse_language_intent(self, transcribed_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse user's language intent from transcribed text using JSON response with retry logic.
        
        Args:
            transcribed_text: The transcribed text from user
        
        Returns:
            Tuple of (language_from, language_to) or (None, None) if unclear
        """
        if self.logger:
            self.logger.log_step("language_intent_request", {"input": transcribed_text})
        
        system_prompt = self.prompts['translation']['language_intent']['system']
        
        # Call with JSON validation enabled - will retry automatically if JSON is invalid
        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=f"User said: {transcribed_text}",
            validate_json=True,
            verbose=False
        )
        
        # Save raw response for debugging
        if self.logger:
            self.logger.save_file(f"language_intent_raw_response_{self.logger.step_counter}.txt", response, mode='text')
        
        # Parse JSON response (should always be valid after retries)
        parsed = self._parse_json_response(response, "language_intent")
        
        if parsed:
            from_lang = parsed.get('from_language', 'unknown')
            to_lang = parsed.get('to_language', 'unknown')
            confidence = parsed.get('confidence', 'unknown')
            
            if self.logger:
                self.logger.log_step("language_intent_parsed", {
                    "from_language": from_lang,
                    "to_language": to_lang,
                    "confidence": confidence,
                    "original_text": parsed.get('original_text', transcribed_text)
                })
            
            # Validate languages
            if from_lang in ["en", "zh"] and to_lang in ["en", "zh"]:
                return from_lang, to_lang
            else:
                if self.logger:
                    self.logger.log_warning("language_intent_invalid", 
                                          f"Invalid languages: {from_lang}, {to_lang}")
                return None, None
        else:
            if self.logger:
                self.logger.log_error("language_intent_parse_failed", "Could not parse JSON response")
            return None, None
    
    def translate_text(self, text: str, from_lang: str, to_lang: str) -> str:
        """
        Translate text from one language to another using JSON response with retry logic.
        
        Args:
            text: Text to translate
            from_lang: Source language code ('en' or 'zh')
            to_lang: Target language code ('en' or 'zh')
        
        Returns:
            Translated text
        """
        if self.logger:
            self.logger.log_step("translation_request", {
                "text": text,
                "from_language": from_lang,
                "to_language": to_lang
            })
        
        from_lang_name = self.language_names[from_lang]
        to_lang_name = self.language_names[to_lang]
        
        
        # Format system prompt with language names
        system_prompt = self.prompts['translation']['translation']['system'].replace('{from_language_name}', from_lang_name).replace('{to_language_name}', to_lang_name)
        system_prompt = system_prompt.replace('{from_language_code}', from_lang).replace('{to_language_code}', to_lang)
        
        
        # Call with JSON validation enabled - will retry automatically if JSON is invalid
        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=text,
            validate_json=True,
            verbose=False
        )
        
        # Save raw response for debugging
        if self.logger:
            self.logger.save_file(f"translation_raw_response_{self.logger.step_counter}.txt", 
                f"From: {from_lang} -> To: {to_lang}\nInput: {text}\n\n{'='*80}\nRaw Response:\n{'='*80}\n\n{response}", 
                mode='text')
        
        # Parse JSON response (should always be valid after retries)
        parsed = self._parse_json_response(response, "translation")
        
        if parsed and isinstance(parsed, dict) and 'translated_text' in parsed:
            translated_text = parsed['translated_text']
            
            # Safely log with error handling
            if self.logger:
                try:
                    self.logger.log_step("translation_complete", {
                        "original_text": parsed.get('original_text', text),
                        "translated_text": translated_text,
                        "from_language": from_lang,
                        "to_language": to_lang
                    })
                except Exception as log_error:
                    self.logger.log_warning("translation_logging_error", f"Failed to log: {str(log_error)}")
            
            return translated_text
        else:
            # This should rarely happen now due to retry logic
            if self.logger:
                self.logger.log_warning("translation_fallback", f"Using raw response as translation. Parsed: {type(parsed)}")
            # Extract just the text from the response if possible
            return response.strip()
    
    def translate_audio_to_audio(
        self, 
        input_audio_path: str, 
        from_lang: str, 
        to_lang: str,
        output_audio_path: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Complete audio-to-audio translation pipeline with logging.
        
        Args:
            input_audio_path: Path to input audio file
            from_lang: Source language code
            to_lang: Target language code
            output_audio_path: Optional path for output audio
        
        Returns:
            Tuple of (transcribed_text, translated_text, output_audio_path)
        """
        try:
            if self.logger:
                self.logger.log_step("audio_translation_start", {
                    "input_audio": input_audio_path,
                    "from_language": from_lang,
                    "to_language": to_lang
                })
            
            # Step 1: Transcribe input audio
            transcribed_text = self.transcribe_audio(input_audio_path)
            
            # Step 2: Translate text
            translated_text = self.translate_text(transcribed_text, from_lang, to_lang)
            
            # Step 3: Generate audio from translation
            if output_audio_path is None:
                fd, output_audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
            
            self.tts.generate_simple(
                text=translated_text,
                voice=self.voice_map[to_lang],
                output_path=output_audio_path
            )
            
            if self.logger:
                self.logger.log_step("audio_translation_complete", {
                    "transcribed_text": transcribed_text,
                    "translated_text": translated_text,
                    "output_audio": output_audio_path
                })
            
            return transcribed_text, translated_text, output_audio_path
        
        except Exception as e:
            if self.logger:
                self.logger.log_error("audio_translation_pipeline_error", str(e))
            raise
    
    def generate_confirmation_prompt(self, from_lang: str, to_lang: str) -> str:
        """
        Generate confirmation prompt in English.
        
        Args:
            from_lang: Source language code
            to_lang: Target language code
        
        Returns:
            Path to confirmation audio file
        """
        from_lang_name = self.language_names[from_lang]
        to_lang_name = self.language_names[to_lang]
        
        text = self.prompts['translation']['greetings']['confirmation'].format(
            from_language_name=from_lang_name,
            to_language_name=to_lang_name
        )
        
        return self.generate_audio_prompt(text, language="en")
    
    def check_confirmation(self, audio_path: str) -> bool:
        """
        Check if user said 'confirm' or similar using JSON response with retry logic.
        
        Args:
            audio_path: Path to audio file with user's response
        
        Returns:
            True if confirmed, False otherwise
        """
        if self.logger:
            self.logger.log_step("confirmation_request", {"audio_path": audio_path})
        
        transcribed = self.transcribe_audio(audio_path)
        
        system_prompt = self.prompts['translation']['confirmation']['system']
        
        # Call with JSON validation enabled - will retry automatically if JSON is invalid
        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=f"User said: {transcribed}",
            validate_json=True,
            verbose=False
        )
        
        # Save raw response for debugging
        if self.logger:
            self.logger.save_file(f"confirmation_raw_response_{self.logger.step_counter}.txt", response, mode='text')
        
        # Parse JSON response (should always be valid after retries)
        parsed = self._parse_json_response(response, "confirmation")
        
        if parsed and 'is_confirmed' in parsed:
            is_confirmed = parsed['is_confirmed']
            
            if self.logger:
                self.logger.log_step("confirmation_result", {
                    "is_confirmed": is_confirmed,
                    "confidence": parsed.get('confidence', 'unknown'),
                    "transcribed_text": parsed.get('transcribed_text', transcribed),
                    "matched_keywords": parsed.get('matched_keywords', [])
                })
            
            return is_confirmed
        else:
            # Fallback to simple keyword matching (should rarely happen now)
            if self.logger:
                self.logger.log_warning("confirmation_fallback", "Using keyword matching")
            
            confirmation_words = ["confirm", "yes", "okay", "ok", "sure", "correct", "right"]
            transcribed_lower = transcribed.lower()
            return any(word in transcribed_lower for word in confirmation_words)


class TranslationSession:
    """Manages a translation session state."""
    
    def __init__(self, pipeline: TranslationPipeline):
        self.pipeline = pipeline
        self.state = "init"  # States: init, waiting_languages, waiting_confirmation, translating
        self.from_lang: Optional[str] = None
        self.to_lang: Optional[str] = None
    
    def reset(self):
        """Reset session state."""
        self.state = "init"
        self.from_lang = None
        self.to_lang = None
        
        if self.pipeline.logger:
            self.pipeline.logger.log_step("session_reset", {"state": "init"})
    
    def start_session(self) -> str:
        """
        Start a new translation session.
        
        Returns:
            Path to initial greeting audio
        """
        self.reset()
        
        if self.pipeline.logger:
            self.pipeline.logger.log_step("session_start", {"state": "waiting_languages"})
        
        text = self.pipeline.prompts['translation']['greetings']['initial']
        self.state = "waiting_languages"
        return self.pipeline.generate_audio_prompt(text, language="en")
