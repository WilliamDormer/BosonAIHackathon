import ast
from typing import Union
import lerobot
import time
import concurrent.futures

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
class RobotWrapper:
    def __init__(self):
        self.valid_keys = {
            'shoulder_pan.pos',
            'shoulder_lift.pos',
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos',
        }

        # entry for each link name (in valid keys), maps to link length (a) 
        self.dh_table ={

        }

        config = SO101FollowerConfig(
            port = '/dev/tty.usbmodem5A7A0558341', # front port on my mac.
            max_relative_target = 5.0, # controls magnitude of relative target vector.
        )

        self.robot = SO101Follower(config=config)
        print("about to call robot.connect")
        # if connection stalls, see if it's plugged in.
        self.robot.connect()

        # setup the robot instance 

    def __call__(self, text):
        # TODO accepts text input (see format) and executes actions on the robot. 

        '''
        This is what the robot accepts: (aka joint space)
        {
            "shoulder_pan.pos": 45.0,
            "shoulder_lift.pos": 30.0,
            "elbow_flex.pos": -10.0,
            "wrist_flex.pos": 5.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 100
        }
        Within the bounds provided by the configuration file (that you generate and save locally on your laptop)
        '''

        # Step 1: Validate that the input text follows the required format. # TODO this might require some parsing or approximation logic.
        d = ast.literal_eval(text)
        assert isinstance(d, dict), f"The provided string could not be parsed as a dictionary: {text}"

        for key, value in d.items():
            assert key in self.valid_keys, f"key found in dict isn't present in valid_keys: {key}"
            assert isinstance(value, int) or isinstance(value, float), f"key: {key} had a value of type: {type(value)}, when int or float is required."
        
        # now the input should be syntactically valid. (values could still be incorrect.)
        
        # Ensure that the chosen action doesn't violate any joint limits. 
        
        
        # Call send action in lerobot.

        _ = self.robot.send_action(d)

    # def __del__(self):
    #     # disconnect the robot.
    #     self.robot.disconnect()


if __name__ == "__main__":
    # test logic for the class.

    # sample_text = "{'shoulder_pan.pos': -45.0,'shoulder_lift.pos': 20.0,'elbow_flex.pos': -10.0,'wrist_flex.pos': 5.0,'wrist_roll.pos': 0.0,'gripper.pos': 100}"
    sample_text = "{'shoulder_pan.pos': 0.0,'shoulder_lift.pos': 0.0,'elbow_flex.pos': -90.0,'wrist_flex.pos': -180.0,'wrist_roll.pos': 0.0,'gripper.pos': 100}"


    # Calibration saved to /Users/williamdormer/.cache/huggingface/lerobot/calibration/robots/so101_follower/None.json

    robot = RobotWrapper()

    for i in range(30):
        robot(sample_text)
        time.sleep(0.1)
