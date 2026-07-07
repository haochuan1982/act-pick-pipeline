#!/usr/bin/env python3
"""
Async Inference Handler - Runs model inference in background thread
"""

import queue
import threading
import time
from typing import Callable, Any, Tuple, Optional
import numpy as np


class AsyncInference:
    """Runs model inference in background thread with callback"""

    def __init__(self, inference_callback: Callable[[Any], Tuple[np.ndarray, float]]):
        """
        Initialize async inference handler.

        Args:
            inference_callback: Function that takes inputs and returns (actions, inference_time_ms)
                              Example: lambda inputs: (model.infer(inputs['state'], inputs['images']), time_ms)
        """
        self.inference_callback = inference_callback
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True

        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.daemon = True
        self.thread.start()

    def _inference_loop(self):
        """Background thread for inference"""
        while self.running:
            try:
                inputs = self.inference_queue.get(timeout=0.1)

                t0 = time.time()
                result = self.inference_callback(inputs)
                inference_time = (time.time() - t0) * 1000

                # Result can be just actions, or (actions, custom_time)
                if isinstance(result, tuple):
                    actions, custom_time = result
                    inference_time = custom_time
                else:
                    actions = result

                # Put result (discard old if exists)
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put((actions, inference_time))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                import traceback
                traceback.print_exc()

    def request_inference(self, inputs: Any) -> bool:
        """
        Request inference (non-blocking, discards old request).

        Args:
            inputs: Input data to pass to inference_callback

        Returns:
            True if request was queued, False otherwise
        """
        try:
            # Clear old request
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                pass
            # Add new request
            self.inference_queue.put_nowait(inputs)
            return True
        except queue.Full:
            return False

    def get_result(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get inference result if ready (non-blocking).

        Returns:
            (actions, inference_time_ms) if result available, None otherwise
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the background inference thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
