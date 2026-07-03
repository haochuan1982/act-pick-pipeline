#!/usr/bin/env python3
"""
ACT Model Inference Pipeline for LeRobot Arm with 3 Cameras
Reads from:
- /dev/video8 (UGREEN Camera - top)
- /dev/video6 (USB2.0_CAM1 - front)
- /dev/video0 (RealSense RGB - head)
- /dev/ttyACM1 (LeRobot SO101 follower arm)
"""

import cv2
import numpy as np
import time
from pathlib import Path

from .act_model import ACTModelInference
from .lerobot_arm_controller import LeRobotArmController


class CameraManager:
    """Manages multiple camera inputs"""

    def __init__(self, camera_mapping):
        """
        camera_mapping: dict like {'top': 8, 'front': 6, 'head': 0}
        """
        self.cameras = {}
        self.camera_mapping = camera_mapping

        for name, device_id in camera_mapping.items():
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {name} at /dev/video{device_id}")

            # Set resolution to 640x480 (will be resized to 384x384 later)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.cameras[name] = cap
            print(f"✓ Opened camera '{name}' at /dev/video{device_id}")

    def read_frames(self):
        """Read frames from all cameras"""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read from camera '{name}'")
            frames[name] = frame
        return frames

    def close(self):
        """Release all cameras"""
        for cap in self.cameras.values():
            cap.release()



def main():
    """Main inference loop"""

    # Configuration
    # Note: RealSense 435i creates multiple /dev/video* interfaces.
    # video0 is metadata channel (can't be opened with OpenCV)
    # video1-5 are actual video streams (RGB, IR, depth, etc.)
    CAMERA_MAPPING = {
        'top': 8,     # UGREEN Camera 1080P
        'front': 6,   # USB2.0_CAM1
        'head': 2,    # RealSense 435i RGB (video1, not video0!)
    }

    ROBOT_PORT = '/dev/ttyACM1'
    MODEL_PATH = 'openvino'  # Relative to pipeline/ directory

    # Control parameters
    ACTION_CHUNK_SIZE = 100  # Execute first N actions from prediction
    CONTROL_FREQUENCY = 30  # Hz

    print("=" * 60)
    print("ACT Inference Pipeline - LeRobot Cube Picking")
    print("=" * 60)

    try:
        # Initialize components
        print("\n[1/3] Initializing cameras...")
        cameras = CameraManager(CAMERA_MAPPING)

        print("\n[2/3] Connecting to robot arm...")
        robot = LeRobotArmController(port=ROBOT_PORT)

        print("\n[3/3] Loading ACT model...")
        model = ACTModelInference(MODEL_PATH, device='CPU')

        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Main inference loop
        action_queue = []
        step_count = 0

        while True:
            loop_start = time.time()

            # Read camera frames
            frames = cameras.read_frames()

            # Read current robot state
            positions = robot.read_joint_positions()

            # Run inference every time action queue is empty
            if len(action_queue) == 0:
                print(f"\n[Step {step_count}] Running inference...")
                print(f"  Current state: {current_state}")

                # Predict action sequence
                actions = model.infer(positions, frames)

                # Queue first N actions
                action_queue = list(actions[:ACTION_CHUNK_SIZE])
                print(f"  Predicted {len(actions)} actions, queued {len(action_queue)}")

            # Execute next action
            if action_queue:
                next_action = action_queue.pop(0)
                print(f"  Executing action {ACTION_CHUNK_SIZE - len(action_queue)}/{ACTION_CHUNK_SIZE}: {next_action}")
                robot.send_joint_positions(next_action)

            step_count += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            time.sleep(sleep_time)

            # Show FPS
            actual_freq = 1.0 / (elapsed + sleep_time)
            print(f"  Frequency: {actual_freq:.1f} Hz", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if 'cameras' in locals():
            cameras.close()
        if 'robot' in locals():
            robot.close()
        print("✓ Done")


if __name__ == '__main__':
    main()
