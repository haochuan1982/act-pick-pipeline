#!/usr/bin/env python3

import asyncio
import time
import numpy as np
from pathlib import Path
from collections import deque
from collections import deque

from lerobot_arm_controller import LeRobotArmController
from camera_utils import build_shared_camera
from model_utils import load_inference_model
from AsyncInference import AsyncInference

from physicalai.capture import SharedCamera
from physicalai.capture.errors import CaptureError
#from physicalai.data import Observation

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
                'name' : "front",
                'driver': "usb_camera",
                'fingerprint': "/dev/video6",
                'hardwarename': "UGreen Camera 1080P",
                'payload': { 'width': 640, 'height': 480, 'fps': 30 },
            },
            {
                'name' : "header",
                'driver': "usb_camera",
                'fingerprint': "/dev/video8",
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
        ROBOT_PORT = '/dev/ttyACM2'
        robot = LeRobotArmController(port=ROBOT_PORT)
        robot.connect()

        # initialize model
        MODEL_CONFIG = {
            'model_path': './openvino',
            'backend': 'openvino',
            'policy_name': 'act',
            'device': 'NPU'
        }
        inference_model = load_inference_model(MODEL_CONFIG)

        # Create async inference handler with callback
        def inference_callback(inputs):
            """Callback for async inference - uses physicalai InferenceModel"""
            output = inference_model.select_action(inputs)
            return output

        async_inference = AsyncInference(inference_callback)

        # run pipeline
        print("\n" + "=" * 60)
        print("✓ System ready! Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        ACTION_CHUNK_SIZE = 100
        CONTROL_FREQUENCY = 10  # Hz
        action_queue = deque()

        for i in range(2):
            # Prepare inputs
            inputs = {}
            obs = robot.read_joint_positions()
            inputs['state'] = obs[np.newaxis]  # Add batch dimension: [6] -> [1, 6]
            for cam_name, cam in frame_captures.items():
                frame = cam.read_latest()
                inputs[cam_name] = np.ascontiguousarray(frame.data[..., ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255)

            # Request async inference
            async_inference.request_inference(inputs)

            # Wait for result (for testing - in production you'd do this differently)
            result = None
            while result is None:
                result = async_inference.get_result()
                if result is None:
                    await asyncio.sleep(0.01)

            output, inference_time = result
            print(f"Inference completed in {inference_time:.1f}ms, output: {output}")

            # TODO: Execute actions on robot
            # robot.send_joint_positions(output)

            time.sleep(1.0 / 30)

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
