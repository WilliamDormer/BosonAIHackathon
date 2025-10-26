import ast
from typing import Union
import lerobot
import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
class RobotWrapper:
    def __init__(self, action_type: str = "joint"):

        self.action_type = action_type

        # always save these for the read_current_position function.
        self.valid_keys_joint = {
            'shoulder_pan.pos',
            'shoulder_lift.pos',
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos',
        }

        # set the valid keys bsaed upon the action type
        if action_type == "joint":
            self.valid_keys = {
                'shoulder_pan.pos',
                'shoulder_lift.pos',
                'elbow_flex.pos',
                'wrist_flex.pos',
                'wrist_roll.pos',
                'gripper.pos',
            }
            
        elif action_type == "end_effector":
            self.valid_keys = {
                'ee.x',
                'ee.y',
                'ee.z',
                'ee.wx',
                'ee.wy',
                'ee.wz',
                'ee.gripper_pos'
            }
        else:
            raise ValueError(f"Unknown action_type provided to RobotWrapper: {action_type}")

        config = SO101FollowerConfig(
            port = '/dev/tty.usbmodem5A7A0558341', # front port on my mac.
            max_relative_target = 5.0, # controls magnitude of relative target vector.
        )

        self.robot = SO101Follower(config=config)
        print("about to call robot.connect")
        # if connection stalls, see if it's plugged in.
        self.robot.connect()

        # setup the robot instance 

    def __call__(self, text : str | dict):
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

        print("call called in RobotWrapper")

        if isinstance(text, dict):
            d = text
        elif isinstance(text, str):
            # Step 1: Validate that the input text follows the required format. # TODO this might require some parsing or approximation logic.
            d = ast.literal_eval(text)
        else: 
            raise ValueError(f"Input to RobotWrapper must be str or dict, got: {type(text)}")
        assert isinstance(d, dict), f"The provided string could not be parsed as a dictionary: {text}"

        for key, value in d.items():
            assert key in self.valid_keys, f"key found in dict isn't present in valid_keys: {key}"
            assert isinstance(value, int) or isinstance(value, float), f"key: {key} had a value of type: {type(value)}, when int or float is required."
        
        # now the input should be syntactically valid. (values could still be incorrect.)
        
        # Ensure that the chosen action doesn't violate any joint limits. 
        
        
        # Call send action in lerobot.

        # send the action until we reach the position.

        if self.action_type == "end_effector":
            d = self.robot.compute_joint_positions_from_ee(d)

        current_pos = self._read_current_position() # in joint space. 
    
        # if d is missing keys, fill in with current_pos keys
        for key in self.valid_keys_joint:
            # fill in the missing keys
            if key not in d:
                d[key] = current_pos[key]


        # print("current_pos: ", current_pos)
        # print("d: ", d)
        # print("type current pos: ", type(current_pos))
        while not self._position_close_enough(target=d, current=current_pos):
            # since d has been put to joint space, we send all actions in joint space.
            _ = self.robot.send_action(d, "joint")
            time.sleep(0.01)
            current_pos = self._read_current_position()
    def _read_current_position(self, space="joint") -> dict:
        # get the position from servos.
        current_pos = self.robot.bus.sync_read("Present_Position")

        # for each key, append.pos to the end of the key name
        pos_dict = {}
        for i, key in enumerate(self.valid_keys_joint):
            # strip .pos suffix
            key_without_pos = key.removesuffix('.pos')
            pos_dict[key] = current_pos[key_without_pos]
        return pos_dict

    def _position_close_enough(self, target: dict, current: dict, threshold: float = 2.0) -> bool:
        """Check if the current position is within the threshold of the target position."""
        for key in target.keys():
            if abs(target[key] - current[key]) > threshold:
                print(f"for key {key}, wasn't close enough because target: {target[key]}, current: {current[key]}")
                return False
        return True


if __name__ == "__main__":
    # test logic for the class.

    # sample_text = "{'shoulder_pan.pos': -45.0,'shoulder_lift.pos': 20.0,'elbow_flex.pos': -10.0,'wrist_flex.pos': 5.0,'wrist_roll.pos': 0.0,'gripper.pos': 100}"
    sample_text = "{'shoulder_pan.pos': 0.0,'shoulder_lift.pos': 0.0,'elbow_flex.pos': 0.0,'wrist_flex.pos': 0.0,'wrist_roll.pos': 0.0,'gripper.pos': 0}"


    sample_ee_text = {
       "ee.x": 0.2,
       "ee.y": 0.0,
       "ee.z": 0.0,
       "ee.wx": 0.0,
       "ee.wy": 0.0,
       "ee.wz": 0.0,
       "ee.gripper_pos": 50.0
    }

    # Calibration saved to /Users/williamdormer/.cache/huggingface/lerobot/calibration/robots/so101_follower/None.json

    robot = RobotWrapper(action_type = "end_effector")
    with robot.robot.bus.torque_disabled():
        while True:
            result = robot._read_current_position()
            print("result: ", result)
            time.sleep(5)

    # for i in range(30):
    # robot(sample_ee_text)
        # time.sleep(0.1)
