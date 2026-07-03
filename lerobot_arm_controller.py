import json
import os
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

    def read_joint_positions(self):
        try:
            # Get observation (includes motor positions)
            obs = self._robot.get_observation()

            if False:
                positions: dict[str, float] = {}
                for i, name in enumerate(self._robot.joint_names):
                    raw_position = float(obs.joint_positions[i])
                    positions[f"{name}.pos"] = raw_position

            return obs

        except Exception as e:
            print(f"Error reading joint positions: {e}")
            return None


    def send_joint_positions(self, positions):
        """
        Send target joint positions to the arm.

        Args:
            positions: np.array of shape (6,) with normalized positions
        """
        try:
            if len(positions) != 6:
                raise ValueError(f"Expected 6 positions, got {len(positions)}")

            if False:
                action = {f"{motor}.pos": float(pos) for motor, pos in zip(self.MOTOR_NAMES, positions)}

            self._robot.send_action(positions)

        except Exception as e:
            print(f"Error sending joint positions: {e}")

    def close(self):
        """Disconnect from robot"""
        try:
            self.robot.disconnect()
            print("✓ Robot disconnected")
        except Exception as e:
            print(f"Error disconnecting: {e}")
