import os
import json
import tempfile
import yaml  # type: ignore
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.llm import LLM  # type: ignore
from models.asr import ASR  # type: ignore
from models.tts import TTS  # type: ignore
from models.mllm import MLLM  # type: ignore
from robot_control import RobotWrapper

# Import utilities
from utils.helpers import (  # type: ignore
    create_text_message,
    create_tool_call_schema
)

class VoiceRobotAgent:
    
    def __init__(self, prompt_system: str, logger,  api_key):
        self.api_key = api_key
        self.logger = logger

        self.robot = RobotWrapper()

        self.prompt_system = prompt_system

        # Load prompts
        self.prompts = self._load_prompts()

        # Initialize models with config.
        self.asr = ASR(api_key_override=self.api_key)

    def _load_prompts(self) -> Dict[str, Any]:
        """load the prompt from yaml file."""

        prompts_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "robot", self.prompt_system)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
        
    def process_input(self, audio_path: str, mode: str) -> None:
        '''
        Process the audio input from the user and execute the robot action.
        mode: the instruction mode. directions, or box.
        '''

        assert audio_path is not None, "audio_path was none."
        self.logger.log_result(
            "ASR", {
                "action" : "audio_transcription_start",
                "content": "Starting audio transcription."
            }
        )

        print("about to call self._transcribe_audio")

        transcription = self._transcribe_audio(audio_path)
        # print("warning in debug mode overriding ")
        # transcription = input("enter the transcription string for testing: ")

        print("done with self._transcribe_audio")

        self.logger.log_result("🎤 ASR", {
            "action": "audio_transcription_complete",
            "content": f"Transcription: '{transcription}'"
        })

        # TODO use that to move the robot.    
        self._execute_action_from_text(transcription, mode=mode)
        print("end of process_input in VoiceRobotAgent")

    def _execute_action_from_text(self, text: str, mode="directions") -> None:
        '''
        Uses the text input to execute an action on the robot. 
        mode: "directions" # can add more options later.
        '''

        print("execute actions from text called with text: ", text)

        if mode == "directions":
            '''
            expects a string that contains one of the following:
            - left
            - right
            - up
            - down
            - open
            - close

            and takes a preset action on the robot based upon this.
            '''
            if text == "left":
                action = {
                    "shoulder_pan.pos":  -45.0,
                }
            elif text == "right":
                action = {
                    "shoulder_pan.pos":  45.0,
                }
            elif text == "up":
                action = {
                    "wrist_flex.pos": -90.0
                }
            elif text == "down":

                action = {
                    "wrist_flex.pos": 75.0
                }

            elif text == "open":

                action = {
                    "gripper.pos": 100.0
                }
            elif text == "close":
                action = {
                    "gripper.pos": 5.0
                }
            else:
                print(f"Unknown direction provided to _execute_action_from_text: {text}")
                action = {}
            
        elif mode == "box" or mode == "approximate_box":
            # this expects the result to be returned as "[x,y,z,gripper]"
            # assume for now the max is 10 in each dimension.
            # they start at the bottom left corner and go to top right (from the robot's forward direction)

            # # define a mapping from the input space x,y,z to the joint positions
            # reference_mapping = {
            #     (0,0,0) : {'shoulder_pan.pos': -78.91373801916933, 'wrist_flex.pos': 17.519514310494372, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': -51.374207188160675, 'gripper.pos': 5.052631578947368, 'elbow_flex.pos': 98.64864864864865},
            #     (10,0,0) : {'shoulder_pan.pos': 55.69755058572949, 'wrist_flex.pos': 17.519514310494372, 'wrist_roll.pos': 1.3431, 'shoulder_lift.pos': -51.289640591966176, 'gripper.pos': 4.982456140350877, 'elbow_flex.pos': 98.64864864864865},
            #     (10,10,0) : {'shoulder_pan.pos': 34.29179978700745, 'wrist_flex.pos': 16.912402428447535, 'wrist_roll.pos': 1.2942612942612897, 'shoulder_lift.pos': 7.653276955602536, 'gripper.pos': 4.912280701754386, 'elbow_flex.pos': 40.92664092664094},
            #     (0,10,0) : {'shoulder_pan.pos': -37.912673056443026, 'wrist_flex.pos': 16.912402428447535, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': 8.498942917547552, 'gripper.pos': 4.982456140350877, 'elbow_flex.pos': 40.15444015444015},
            #     (5,5,10) : {'shoulder_pan.pos': -8.413205537806178, 'wrist_flex.pos': 17.519514310494372, 'elbow_flex.pos': 35.61776061776061, 'gripper.pos': 5.052631578947368, 'wrist_roll.pos': 1.245421245421241, 'shoulder_lift.pos': -50.69767441860465},
            # (10,10,10): {'shoulder_pan.pos': 32.90734824281151, 'wrist_flex.pos': 12.22896790980053, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': -28.541226215644826, 'gripper.pos': 10.175438596491228, 'elbow_flex.pos': 30.501930501930502},
            #     (0,10,10) : {'shoulder_pan.pos': -42.917997870074544, 'wrist_flex.pos': 12.22896790980053, 'wrist_roll.pos': 1.2942612942612897, 'shoulder_lift.pos': -27.780126849894287, 'gripper.pos': 10.175438596491228, 'elbow_flex.pos': 34.555984555984566},
            #     (10,0,10) : {'shoulder_pan.pos': 45.68690095846645, 'wrist_flex.pos': 67.64960971379011, 'wrist_roll.pos': 1.0012210012209977, 'shoulder_lift.pos': -73.69978858350952,'gripper.pos': 10.105263157894736, 'elbow_flex.pos': 26.64092664092663 },
            #     (0,0,10) : {'shoulder_pan.pos': -58.466453674121404, 'wrist_flex.pos': 67.64960971379011, 'wrist_roll.pos': 1.0012210012209977, 'shoulder_lift.pos': -78.8583509513742,'gripper.pos': 10.035087719298245, 'elbow_flex.pos': 29.054054054054063 },
            # }

            reference_mapping = {
                (0,0,0) : {'shoulder_pan.pos': -79.0, 'wrist_flex.pos': 17.519514310494372, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': -51.374207188160675, 'gripper.pos': 5.052631578947368, 'elbow_flex.pos': 98.64864864864865},
                (10,0,0) : {'shoulder_pan.pos': 79.0, 'wrist_flex.pos': 17.519514310494372, 'wrist_roll.pos': 1.3431, 'shoulder_lift.pos': -51.289640591966176, 'gripper.pos': 4.982456140350877, 'elbow_flex.pos': 98.64864864864865},
                (10,10,0) : {'shoulder_pan.pos': 38, 'wrist_flex.pos': 16.912402428447535, 'wrist_roll.pos': 1.2942612942612897, 'shoulder_lift.pos': 7.653276955602536, 'gripper.pos': 4.912280701754386, 'elbow_flex.pos': 40.92664092664094},
                (0,10,0) : {'shoulder_pan.pos': -38, 'wrist_flex.pos': 16.912402428447535, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': 8.498942917547552, 'gripper.pos': 4.982456140350877, 'elbow_flex.pos': 40.15444015444015},
                
                (5,5,10) : {'shoulder_pan.pos': 0.0, 'wrist_flex.pos': 17.519514310494372, 'wrist_roll.pos': 1.245421245421241, 'shoulder_lift.pos': -50.69767441860465, 'gripper.pos': 5.052631578947368, 'elbow_flex.pos': 35.61776061776061},
                (5,5,5) : {'shoulder_pan.pos': 0.0, 'wrist_flex.pos': 6.7649609713790255,'wrist_roll.pos': 1.2942612942612897,  'shoulder_lift.pos': -74.29175475687103, 'gripper.pos': 4.140350877192982, 'elbow_flex.pos': 83.97683397683397},
                
                (10,10,10): {'shoulder_pan.pos': 32.90734824281151, 'wrist_flex.pos': 12.22896790980053, 'wrist_roll.pos': 1.3431013431013383, 'shoulder_lift.pos': -28.541226215644826, 'gripper.pos': 10.175438596491228, 'elbow_flex.pos': 30.501930501930502}, # top front right
                (0,10,10) : {'shoulder_pan.pos': -42.917997870074544, 'wrist_flex.pos': 12.22896790980053, 'wrist_roll.pos': 1.2942612942612897, 'shoulder_lift.pos': -27.780126849894287, 'gripper.pos': 10.175438596491228, 'elbow_flex.pos': 34.555984555984566}, # top front left
                (10,0,10) : {'shoulder_pan.pos': 45.68690095846645, 'wrist_flex.pos': 67.64960971379011, 'wrist_roll.pos': 1.0012210012209977, 'shoulder_lift.pos': -73.69978858350952,'gripper.pos': 10.105263157894736, 'elbow_flex.pos': 26.64092664092663 }, # top back right
                (0,0,10) : {'shoulder_pan.pos': -58.466453674121404, 'wrist_flex.pos': 67.64960971379011, 'wrist_roll.pos': 1.0012210012209977, 'shoulder_lift.pos': -78.8583509513742,'gripper.pos': 10.035087719298245, 'elbow_flex.pos': 29.054054054054063 }, # top back left
            }

            # pos = json.loads(text) # already a list
            pos = text
            grip = pos[3]
            pos = np.array(pos, np.int32)
            pos = pos[:3] # all but the last element.

            # interpolate between the elements
            points = np.array(list(reference_mapping.keys()), dtype=np.float32)
            joint_dicts = list(reference_mapping.values())
            joint_keys = list(joint_dicts[0].keys())

            # Build arrays for each joint key
            joint_arrays = {k: np.array([d[k] for d in joint_dicts], dtype=np.float32) for k in joint_keys}

            print("points: ", points)
            print("joint_arrays: ", joint_arrays)

            # Interpolate for each joint
            interpolated = {}
            for k in joint_keys:
                interp = LinearNDInterpolator(points, joint_arrays[k])
                interpolated[k] = float(interp(pos))

            # Set gripper position
            interpolated["gripper.pos"] = float(grip*10.0)

            action = interpolated

            print("action: ", action)
        else:
            raise ValueError(f"Unknown mode provided to _execute_action_from_text: {mode}")


        self.logger.log_result("action", action)
        # once we have action, call the robot until the position 
        if action != {}:
            self.robot(action)

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        asr_prompt = self.prompts.get("tool_prompts", {}).get("asr", "")

        print("using asr prompt: ", asr_prompt)
        if asr_prompt == "":
            raise ValueError("ASR prompt not found in prompts.")
        
        result = self.asr.call(audio=audio_path, system_prompt=asr_prompt, validate_json=True)
        # parse the JSON response
        try:
            import json
            parsed_result = json.loads(result)
            transcription = parsed_result.get("response", "")
        except (json.JSONDecodeError, KeyError):
            # Fallback to raw result if JSON parsing fails
            print(f"[ASR] Error parsing JSON response: {result}, treating it as a string")
            transcription = result

        # Log input:
        # Log user input
        if self.logger:
            logger_text = f"🤫 user\nTranscription: {transcription}\nAudio file: {os.path.basename(audio_path)}"
            logger_payload = {
                "action": "audio_input",
                "content": transcription,
                "audio_file": os.path.basename(audio_path)
            }
            
            self.logger.log_result(logger_text, logger_payload)
        
        return transcription









        
        

        