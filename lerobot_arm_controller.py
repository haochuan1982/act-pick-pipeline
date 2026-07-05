import json
import os
import numpy as np
import threading
from physicalai.robot.so101 import SO101, SO101Calibration
from physicalai.robot.interface import Robot, RobotObservation

class LeRobotArmController:
    """Controls LeRobot SO101 follower arm using LeRobot SDK"""

    # Motor names in order
    MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, port='/dev/ttyACM1', max_relative_target=None):
        self.port = port

        # Load calibration from robot_calibration.json
        calibration_path = os.path.join(os.path.dirname(__file__), 'robot_calibration.json')
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)

        so101_cal = SO101Calibration.from_dict(
            {
                name: {
                    "id": val["id"],
                    "drive_mode": val["drive_mode"],
                    "homing_offset": val["homing_offset"],
                    "range_min": val["range_min"],
                    "range_max": val["range_max"],
                }
                for name, val in calibration.items()
            }
        )

        self._robot = SO101(port=port, calibration=so101_cal, role="follower", unit="normalized")

    def read_joint_positions(self, timeout=0.5):
        """Read joint positions with timeout"""
        result = [None]  # Store result
        error = [None]   # Store error

        def read_with_timeout():
            try:
                obs = self._robot.get_observation()
                joint_positions = np.array(obs.joint_positions, dtype=np.float32)
                result[0] = joint_positions
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=read_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print(f"Warning: Robot read timeout after {timeout}s")
            return None

        if error[0] is not None:
            print(f"Error reading joint positions: {error[0]}")
            return None

        return result[0]


    def send_joint_positions(self, positions, timeout=0.5):
        """Send joint positions with timeout"""
        error = [None]

        def send_with_timeout():
            try:
                if len(positions) != 6:
                    raise ValueError(f"Expected 6 positions, got {len(positions)}")

                self._robot.send_action(positions)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=send_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print(f"Warning: Robot send timeout after {timeout}s")
            return False

        if error[0] is not None:
            print(f"Error sending joint positions: {error[0]}")
            return False

        return True

    def connect(self):
        try:
            self._robot.connect()
            print("✓ Robot connected")
        except Exception as e:
            print(f"Error connecting: {e}")

    def disconnect(self):
        try:
            self._robot.disconnect()
            print("✓ Robot disconnected")
        except Exception as e:
            print(f"Error disconnecting: {e}")

