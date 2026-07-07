#!/usr/bin/env python3
"""
ACT Model Inference Pipeline for LeRobot Arm with 3 Cameras
Optimized version with async operations and hang prevention
"""

import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from collections import deque

from act_model import ACTModelInference
from lerobot_arm_controller import LeRobotArmController
from AsyncInference import AsyncInference


class CameraManager:
    """Manages multiple camera inputs with parallel reading and buffering"""

    def __init__(self, camera_mapping):
        self.cameras = {}
        self.camera_mapping = camera_mapping
        self.frame_queues = {}
        self.reader_threads = {}
        self.running = True

        for name, device_id in camera_mapping.items():
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {name} at /dev/video{device_id}")

            # Reduce buffer size to get fresher frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.cameras[name] = cap
            self.frame_queues[name] = queue.Queue(maxsize=2)

            # Start background reader thread for each camera
            thread = threading.Thread(target=self._read_camera_loop, args=(name, cap))
            thread.daemon = True
            thread.start()
            self.reader_threads[name] = thread

            print(f"✓ Opened camera '{name}' at /dev/video{device_id}")

    def _read_camera_loop(self, name, cap):
        """Background thread continuously reads from camera"""
        consecutive_failures = 0

        while self.running:
            try:
                ret, frame = cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print(f"Camera {name}: too many failures, trying to reopen...")
                        cap.release()
                        time.sleep(1)
                        device_id = self.camera_mapping[name]
                        cap = cv2.VideoCapture(device_id)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                # Put frame in queue (non-blocking, discard if full to keep fresh)
                try:
                    self.frame_queues[name].put_nowait((time.time(), frame))
                except queue.Full:
                    # Discard oldest frame and add new one
                    try:
                        self.frame_queues[name].get_nowait()
                        self.frame_queues[name].put_nowait((time.time(), frame))
                    except:
                        pass

            except Exception as e:
                print(f"Camera {name} error: {e}")
                time.sleep(0.1)

    def read_frames(self, max_age=0.1):
        """Get latest frames from all cameras (non-blocking)"""
        frames = {}

        for name in self.cameras.keys():
            try:
                # Get most recent frame from queue
                timestamp, frame = self.frame_queues[name].get(timeout=0.1)

                # Check if frame is too old
                age = time.time() - timestamp
                if age > max_age:
                    print(f"Warning: Camera {name} frame is {age*1000:.0f}ms old")

                frames[name] = frame

            except queue.Empty:
                print(f"Warning: No frame available from camera {name}")
                return None

        return frames

    def close(self):
        """Stop all reader threads and release cameras"""
        self.running = False
        time.sleep(0.2)  # Give threads time to stop

        for cap in self.cameras.values():
            cap.release()


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
