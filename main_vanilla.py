#!/usr/bin/env python3
"""
ACT Model Inference Pipeline for LeRobot Arm with 3 Cameras
Vanilla version with OpenCV-based camera management
"""

import numpy as np
import time
import queue
import threading
from pathlib import Path
from collections import deque

from ActModel import ACTModelInference
from CameraManager import CameraManager
from async_inference import AsyncInference

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main():
    """Main inference loop"""

    # Configuration
    CAMERA_MAPPING = {
        'top': 2,
        'front': 6,
        'head': 8,
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
        robot_config = SO101FollowerConfig(
            port=ROBOT_PORT,
            max_relative_target=None,
            use_degrees=True
        )
        robot = SO101Follower(robot_config)
        robot.connect(calibrate=True)  # Skip calibration if already calibrated

        print("\n[3/3] Loading ACT model...")
        model = ACTModelInference(MODEL_PATH, device='NPU')

        # Initialize result queue and shared state
        result_queue = queue.Queue()
        current_inputs = {}
        inference_busy = [False]

        ACTION_QUEUE_THRESHOLD = 3  # Trigger when queue has fewer than 3 actions

        def run_inference():
            """Run inference and add actions to queue"""
            if result_queue.qsize() < ACTION_QUEUE_THRESHOLD and not inference_busy[0] and current_inputs:
                inference_busy[0] = True
                t0 = time.time()

                state, images = current_inputs['state'], current_inputs['images']
                actions = model.infer(state, images)
                inference_time = (time.time() - t0) * 1000

                # Add actions to result queue
                for action in actions:
                    result_queue.put(action)

                print(f"Inference: {inference_time:.1f}ms, added {len(actions)} actions, queue size: {result_queue.qsize()}")
                inference_busy[0] = False

        async_inference = AsyncInference(run_inference)

        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Main control loop
        step_count = 0

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
            try:
                observation = robot.get_observation()
                # Extract position values from observation dict (keys like "shoulder_pan.pos", etc.)
                motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                positions = np.array([observation[f"{motor}.pos"] for motor in motor_names], dtype=np.float32)
                t_robot_read = (time.time() - t0) * 1000
            except Exception as e:
                print(f"Warning: Robot read failed: {e}")
                time.sleep(0.01)
                continue

            # Update current inputs for inference
            current_inputs = {'state': positions, 'images': frames}

            # Get action from result queue (non-blocking)
            try:
                action = result_queue.get(timeout=0.1)
            except queue.Empty:
                print("Warning: No action available, skipping iteration")
                time.sleep(0.01)
                continue

            # Execute action
            t0 = time.time()
            try:
                # Convert action array to RobotAction dict format
                motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                action_dict = {f"{motor}.pos": float(action[i]) for i, motor in enumerate(motor_names)}
                robot.send_action(action_dict)
                t_robot_write = (time.time() - t0) * 1000
            except Exception as e:
                print(f"Warning: Robot write failed: {e}")
                t_robot_write = 0

            step_count += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            time.sleep(sleep_time)

            # Show status
            if step_count % 30 == 0:  # Print every 1 second
                total_time = (elapsed + sleep_time) * 1000
                actual_freq = 1.0 / (elapsed + sleep_time)
                status = f"[Step {step_count}] Loop: {total_time:.1f}ms ({actual_freq:.1f}Hz) | "
                status += f"cam:{t_camera:.1f}ms robot:{t_robot_read:.1f}ms | "
                status += f"queue:{result_queue.qsize()}"
                print(status)

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
