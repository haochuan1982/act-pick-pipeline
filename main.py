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
import threading
from pathlib import Path

from act_model import ACTModelInference
from lerobot_arm_controller import LeRobotArmController


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

    def read_frames(self, timeout=1.0):
        """Read frames from all cameras with timeout"""
        frames = {}

        for name, cap in self.cameras.items():
            # Use threading to implement timeout
            result = [None, None]  # [ret, frame]

            def read_with_timeout():
                result[0], result[1] = cap.read()

            thread = threading.Thread(target=read_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                # Timeout occurred
                print(f"Warning: Camera '{name}' read timeout after {timeout}s")
                return None

            ret, frame = result
            if not ret:
                print(f"Warning: Camera '{name}' read failed")
                return None

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

    ROBOT_PORT = '/dev/ttyACM2'
    MODEL_PATH = 'openvino'  # Relative to pipeline/ directory

    # Control parameters
    ACTION_CHUNK_SIZE = 100  # Execute first N actions from prediction
    CONTROL_FREQUENCY = 10  # Hz

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
        model = ACTModelInference(MODEL_PATH, device='NPU')

        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Main inference loop
        action_queue = []
        step_count = 0

        robot.connect()

        while True:
            loop_start = time.time()

            # Read camera frames with timeout
            t0 = time.time()
            frames = cameras.read_frames(timeout=0.5)  # 500ms timeout per camera
            t_camera = (time.time() - t0) * 1000  # ms

            # Skip this iteration if camera read failed/timeout
            if frames is None:
                print("Warning: Camera read failed or timeout, skipping iteration")
                time.sleep(0.01)  # Brief pause before retry
                continue

            # Read current robot state
            t0 = time.time()
            positions = robot.read_joint_positions()
            t_robot_read = (time.time() - t0) * 1000  # ms

            # Skip this iteration if robot read failed
            if positions is None:
                print("Warning: Failed to read robot state, skipping iteration")
                continue

            # Run inference every time action queue is empty
            if len(action_queue) == 0:
                print(f"\n[Step {step_count}] Running inference...")
                print(f"  Current state: {positions}")

                # Predict action sequence
                t0 = time.time()
                actions = model.infer(positions, frames)
                t_inference = (time.time() - t0) * 1000  # ms

                # Queue first N actions
                action_queue = list(actions[:ACTION_CHUNK_SIZE])
                print(f"  Predicted {len(actions)} actions, queued {len(action_queue)}")
                print(f"  Timing: camera={t_camera:.1f}ms, robot_read={t_robot_read:.1f}ms, inference={t_inference:.1f}ms")

            # Execute next action
            if action_queue:
                t0 = time.time()
                next_action = action_queue.pop(0)
                robot.send_joint_positions(next_action)
                time.sleep(0.001)
                t_robot_write = (time.time() - t0) * 1000  # ms
                #print(f"  Executing action {ACTION_CHUNK_SIZE - len(action_queue)}/{ACTION_CHUNK_SIZE} [took {t_robot_write:.1f}ms]")

            step_count += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            time.sleep(sleep_time)

            # Show FPS and timing
            total_time = (elapsed + sleep_time) * 1000  # ms
            actual_freq = 1.0 / (elapsed + sleep_time)
            #print(f"  Loop: {total_time:.1f}ms ({actual_freq:.1f} Hz) | camera={t_camera:.1f}ms robot={t_robot_read:.1f}ms", end='\r')

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
            robot.disconnect()
        print("✓ Done")


if __name__ == '__main__':
    main()
