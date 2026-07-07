#!/usr/bin/env python3

import threading
import time
from typing import Callable


class AsyncInference:

    def __init__(self, inference_callback: Callable[[], None]):
        """
        Initialize async inference handler.
        """

        self.inference_callback = inference_callback
        self.running = True

        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.daemon = True
        self.thread.start()

    def _inference_loop(self):
        """Background thread that repeatedly calls the inference callback"""
        while self.running:
            try:
                self.inference_callback()
                time.sleep(0.01)  # Small sleep to avoid busy-waiting
            except Exception as e:
                print(f"Inference callback error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def stop(self):
        """Stop the background inference thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
