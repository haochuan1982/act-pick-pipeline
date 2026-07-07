#!/usr/bin/env python3
"""
Camera Manager - OpenCV-based camera management with parallel reading and buffering
"""

import cv2
import time
import threading
import queue


class CameraManager:
    """Manages multiple camera inputs with parallel reading and buffering using OpenCV"""

    def __init__(self, camera_mapping):
        """
        Initialize camera manager.

        Args:
            camera_mapping: Dict mapping camera name to device ID
                           Example: {'top': 8, 'front': 6, 'head': 2}
        """
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
        """
        Get latest frames from all cameras (non-blocking).

        Args:
            max_age: Maximum allowed frame age in seconds

        Returns:
            Dict mapping camera name to frame, or None if any camera failed
        """
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
