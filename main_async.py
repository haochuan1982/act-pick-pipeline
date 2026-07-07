#!/usr/bin/env python3
"""
ACT Model Inference Pipeline for LeRobot Arm with 3 Cameras (Async version)
Uses physicalai async camera reading for better performance.

Reads from:
- /dev/video8 (UGREEN Camera - top)
- /dev/video6 (USB2.0_CAM1 - front)
- /dev/video2 (RealSense RGB - head)
- /dev/ttyACM2 (LeRobot SO101 follower arm)
"""

import asyncio
import time
from pathlib import Path

from act_model import ACTModelInference
from lerobot_arm_controller import LeRobotArmController
from CameraManager import CameraManager


async def main_async():
    """Async main inference loop with concurrent camera reading."""

    # Configuration
    CAMERA_MAPPING = {
        'top': 8,     # UGREEN Camera 1080P
        'front': 6,   # USB2.0_CAM1
        'head': 2,    # RealSense 435i RGB (video2, not video0!)
    }

    ROBOT_PORT = '/dev/ttyACM2'
    MODEL_PATH = 'openvino'  # Relative to pipeline/ directory

    # Control parameters
    ACTION_CHUNK_SIZE = 100  # Execute first N actions from prediction
    CONTROL_FREQUENCY = 10  # Hz

    print("=" * 60)
    print("ACT Inference Pipeline - LeRobot Cube Picking (Async)")
    print("=" * 60)

    cameras = None
    robot = None
    model = None

    try:
        # Initialize components
        print("\n[1/3] Initializing cameras...")
        cameras = CameraManager(CAMERA_MAPPING)
        await cameras.async_connect()

        print("\n[2/3] Connecting to robot arm...")
        robot = LeRobotArmController(port=ROBOT_PORT)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, robot.connect)

        print("\n[3/3] Loading ACT model...")
        model = ACTModelInference(MODEL_PATH, device='NPU')

        print("\n" + "=" * 60)
        print("✓ System ready! Starting async inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Main inference loop
        action_queue = []
        step_count = 0

        while True:
            loop_start = time.time()

            # Read camera frames asynchronously
            t0 = time.time()
            frames = await cameras.async_read_frames(timeout=0.5)
            t_camera = (time.time() - t0) * 1000  # ms

            # Skip this iteration if camera read failed/timeout
            if frames is None:
                print("Warning: Camera read failed or timeout, skipping iteration")
                await asyncio.sleep(0.01)  # Brief pause before retry
                continue

            # Read current robot state (offloaded to executor)
            t0 = time.time()
            positions = await loop.run_in_executor(None, robot.read_joint_positions)
            t_robot_read = (time.time() - t0) * 1000  # ms

            # Skip this iteration if robot read failed
            if positions is None:
                print("Warning: Failed to read robot state, skipping iteration")
                continue

            # Run inference every time action queue is empty
            if len(action_queue) == 0:
                print(f"\n[Step {step_count}] Running inference...")
                print(f"  Current state: {positions}")

                # Predict action sequence (offloaded to executor)
                t0 = time.time()
                actions = await loop.run_in_executor(None, model.infer, positions, frames)
                t_inference = (time.time() - t0) * 1000  # ms

                # Queue first N actions
                action_queue = list(actions[:ACTION_CHUNK_SIZE])
                print(f"  Predicted {len(actions)} actions, queued {len(action_queue)}")
                print(f"  Timing: camera={t_camera:.1f}ms, robot_read={t_robot_read:.1f}ms, inference={t_inference:.1f}ms")

            # Execute next action
            if action_queue:
                t0 = time.time()
                next_action = action_queue.pop(0)
                await loop.run_in_executor(None, robot.send_joint_positions, next_action)
                await asyncio.sleep(0.001)
                t_robot_write = (time.time() - t0) * 1000  # ms

            step_count += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            await asyncio.sleep(sleep_time)

            # Show FPS and timing
            total_time = (elapsed + sleep_time) * 1000  # ms
            actual_freq = 1.0 / (elapsed + sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if cameras is not None:
            await cameras.async_close()
        if robot is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, robot.disconnect)
        print("✓ Done")


def main():
    """Entry point that runs the async main loop."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
