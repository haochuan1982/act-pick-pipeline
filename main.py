#!/usr/bin/env python3

import queue
import threading
import asyncio
import time
import numpy as np
from pathlib import Path
from collections import deque

from lerobot_arm_controller import LeRobotArmController
from camera_utils import build_shared_camera
from model_utils import load_inference_model
from async_inference import AsyncInference

from physicalai.capture import SharedCamera
from physicalai.capture.errors import CaptureError
#from physicalai.data import Observation

debug = True

async def main_async():    
    print("=" * 60)
    print("ACT Inference Pipeline - LeRobot Cube Picking")
    print("=" * 60)

    try:
        # initialize cameras
        frame_captures: dict[str, SharedCamera] = {}
        CAMERA_CONFIGS = [
            {
                'name' : "top",
                'driver': "realsense",
                "serial_number": "261222078668",
                'hardwarename': "Intel RealSense D435I",
                'payload': { 'width': 640, 'height': 480, 'fps': 30 },
            },
            {
                'name' : "right",
                'driver': "usb_camera",
                'fingerprint': "/dev/video6",
                'hardwarename': "UGreen Camera 1080P",
                'payload': { 'width': 640, 'height': 480, 'fps': 30 },
            },
            {
                'name' : "left",
                'driver': "usb_camera",
                'fingerprint': "/dev/video8",
                'hardwarename': "UGreen Camera 1080P",
                'payload': { 'width': 640, 'height': 480, 'fps': 30 },
            },
            {
                'name' : "head",
                'driver': "usb_camera",
                'fingerprint': "/dev/video10",
                'hardwarename': "USB 2.0 Camera1",
                'payload': { 'width': 640, 'height': 480, 'fps': 30 },
            }
        ]

        loop = asyncio.get_running_loop()
        for cam_cfg in CAMERA_CONFIGS:
            cam = build_shared_camera(
                    config=cam_cfg,
                    validate_on_connect=True,
                    overwrite_settings=True,
            )
            frame_captures['images.' + cam_cfg['name']] = cam
            print(f"Camera {cam_cfg['name']} initialized")

            try:
                await loop.run_in_executor(None, cam.connect)
            except CaptureError as exc:
                print(f"Camera {cam_cfg['name']}: failed to acquire with requested config: {exc}")
                raise
            await asyncio.sleep(1)  # sleep for camera warmup

        # initialize robot
        #ROBOT_PORT = '/dev/ttyACM1'
        #robot = LeRobotArmController(port=ROBOT_PORT)
        #robot.connect()

        # initialize model
        MODEL_CONFIG = {
            'model_path': './openvino',
            'backend': 'openvino',
            'policy_name': 'act',
            'device': 'GPU'
        }
        inference_model = load_inference_model(MODEL_CONFIG)

        # initialize async inference handler
        result_queue = queue.Queue()
        current_inputs = {}
        inference_busy = [False]  # Use list to allow modification in lambda

        CONTROL_FREQUENCY = 30  # Hz
        ACTION_QUEUE_THRESHOLD = 1

        def run_inference():
            """Run inference and add actions to queue"""
            if result_queue.qsize() < ACTION_QUEUE_THRESHOLD and not inference_busy[0] and current_inputs:

                inference_busy[0] = True
                t0 = time.time()

                actions = inference_model.predict_action_chunk(current_inputs.copy())
                inference_time = (time.time() - t0) * 1000

                for action in actions:
                    result_queue.put(action)

                debug and print(f"Inference time: {inference_time:.1f}ms, added {len(actions)} actions, new queue size: {result_queue.qsize()}")
                inference_busy[0] = False

        async_inference = AsyncInference(run_inference)

        # run pipeline
        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        step_count = 0

        while True:
            loop_start = time.time()
            inputs = {}

            obs = robot.read_joint_positions()
            inputs['state'] = obs[np.newaxis]  # Add batch dimension: [6] -> [1, 6]
            for cam_name, cam in frame_captures.items():
                frame = cam.read_latest()
                inputs[cam_name] = np.ascontiguousarray(frame.data[..., ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255)

            current_inputs = inputs

            try:
                action = result_queue.get(timeout=1.0)
            except queue.Empty:
                print("Warning: No action available, skipping iteration")
                await asyncio.sleep(0.01)
                continue

            robot.send_joint_positions(action)

            step_count += 1
            if debug and step_count % 30 == 0:  # Print every 1 second
                print(f"[Step {step_count}] Action queue size: {result_queue.qsize()}")

            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / CONTROL_FREQUENCY) - elapsed)
            await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if 'async_inference' in locals():
            async_inference.stop()
        if 'frame_captures' in locals():
            for cam in frame_captures.values():
                cam.disconnect()
        if 'robot' in locals():
            robot.disconnect()
        print("✓ Done")


def main():
    """Entry point that runs the async main loop."""
    asyncio.run(main_async())

if __name__ == '__main__':
    main()
