#!/usr/bin/env python3
"""
ACT Model Inference Pipeline for LeRobot Arm with 3 Cameras
Vanilla version with OpenCV-based camera management
"""

import numpy as np
import time
from pathlib import Path
from collections import deque

from ActModel import ACTModelInference
from CameraManager import CameraManager
from lerobot_arm_controller import LeRobotArmController
from async_inference import AsyncInference


def main():
    """Main inference loop"""

    # Configuration
    CAMERA_MAPPING = {
        'top': 8,
        'front': 6,
        'head': 2,
    }

    ROBOT_PORT = '/dev/ttyACM2'
    MODEL_PATH = 'openvino'

    # Control parameters
    ACTION_CHUNK_SIZE = 10
    CONTROL_FREQUENCY = 30  # Hz

    print("=" * 60)
    print("ACT Inference Pipeline - LeRobot Cube Picking (Optimized)")
    print("=" * 60)

    cameras = None
    robot = None
    async_inference = None

    try:
        # Initialize components
        print("\n[1/3] Initializing cameras...")
        cameras = CameraManager(CAMERA_MAPPING)
        time.sleep(0.5)  # Let camera threads start

        print("\n[2/3] Connecting to robot arm...")
        robot = LeRobotArmController(port=ROBOT_PORT)
        robot.connect()

        print("\n[3/3] Loading ACT model...")
        model = ACTModelInference(MODEL_PATH, device='CPU')

        # Create async inference handler with callback
        def inference_callback(inputs):
            """Callback for async inference"""
            state, images = inputs
            actions = model.infer(state, images)
            return actions

        async_inference = AsyncInference(inference_callback)

        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Main control loop
        action_queue = deque()
        step_count = 0
        inference_pending = False
        last_inference_time = 0

        while True:
            loop_start = time.time()

            # Read camera frames (non-blocking)
            t0 = time.time()
            frames = cameras.read_frames(max_age=0.1)
            t_camera = (time.time() - t0) * 1000

            if frames is None:
                print("Warning: Camera read failed, skipping iteration")
                time.sleep(0.01)
                continue

            # Read current robot state
            t0 = time.time()
            positions = robot.read_joint_positions(timeout=0.1)
            t_robot_read = (time.time() - t0) * 1000

            if positions is None:
                print("Warning: Robot read failed, skipping iteration")
                time.sleep(0.01)
                continue

            # Check for inference result
            result = async_inference.get_result()
            if result is not None:
                actions, inference_time = result
                action_queue = deque(actions[:ACTION_CHUNK_SIZE])
                inference_pending = False
                print(f"\n[Step {step_count}] Inference complete: {inference_time:.1f}ms")
                print(f"  Queued {len(action_queue)} actions")

            # Request new inference if queue is low and no inference pending
            if len(action_queue) < 3 and not inference_pending:
                if async_inference.request_inference((positions, frames)):
                    inference_pending = True
                    print(f"  Requested new inference (queue size: {len(action_queue)})")

            # Execute next action
            if action_queue:
                t0 = time.time()
                next_action = action_queue.popleft()
                success = robot.send_joint_positions(next_action, timeout=0.1)
                t_robot_write = (time.time() - t0) * 1000

                if not success:
                    print("Warning: Robot write failed")

            step_count += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            time.sleep(sleep_time)

            # Show status
            total_time = (elapsed + sleep_time) * 1000
            actual_freq = 1.0 / (elapsed + sleep_time)
            status = f"Loop: {total_time:.1f}ms ({actual_freq:.1f}Hz) | "
            status += f"cam:{t_camera:.1f}ms robot:{t_robot_read:.1f}ms | "
            status += f"queue:{len(action_queue)} "
            if inference_pending:
                status += "[INF]"
            print(status, end='\r')

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if async_inference:
            async_inference.stop()
        if cameras:
            cameras.close()
        if robot:
            robot.disconnect()
        print("✓ Done")


if __name__ == '__main__':
    main()
