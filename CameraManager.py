#!/usr/bin/env python3
"""
Camera Manager using physicalai ShapedCamera for async IO.
Based on OpenVINO physicalai capture module.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from physicalai.capture import CameraType, ColorMode, SharedCamera

from physicalai.capture.cameras.uvc import UVCCamera
from physicalai.capture.multi import async_read_cameras
from physicalai.capture.errors import CaptureError, CaptureTimeoutError

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


class CameraManager:
    """Manages multiple camera inputs using physicalai async capture."""

    def __init__(self, camera_mapping: dict[str, int | str]):
        """
        Initialize camera manager with device mapping.

        Args:
            camera_mapping: dict like {'top': 8, 'front': 6, 'head': 2}
                where keys are camera names and values are device IDs
        """
        self.cameras: dict[str, UVCCamera] = {}
        self.camera_mapping = camera_mapping
        self._connected = False

        # Create camera instances (not yet connected)
        for name, device_id in camera_mapping.items():
            self.cameras[name] = UVCCamera(
                device=device_id,
                width=640,
                height=480,
                fps=30,
                color_mode=ColorMode.RGB,
                backend="v4l2",  # Use V4L2 backend on Linux for better performance
            )
            print(f"Created camera instance '{name}' for /dev/video{device_id}")

    def connect(self, timeout: float = 5.0) -> None:
        """
        Connect to all cameras synchronously.

        Args:
            timeout: Maximum seconds to wait for each camera to connect

        Raises:
            CaptureError: If any camera fails to connect
        """
        for name, camera in self.cameras.items():
            try:
                camera.connect(timeout=timeout)
                print(f"✓ Connected to camera '{name}' at {camera.device_id}")
            except CaptureError as e:
                raise RuntimeError(
                    f"Failed to connect camera '{name}' at {camera.device_id}: {e}"
                ) from e

        self._connected = True

    async def async_connect(self, timeout: float = 5.0) -> None:
        """
        Connect to all cameras asynchronously.

        Args:
            timeout: Maximum seconds to wait for each camera to connect
        """
        loop = asyncio.get_running_loop()

        async def connect_one(name: str, camera: UVCCamera) -> None:
            try:
                await loop.run_in_executor(None, camera.connect, timeout)
                print(f"✓ Connected to camera '{name}' at {camera.device_id}")
            except CaptureError as e:
                raise RuntimeError(
                    f"Failed to connect camera '{name}' at {camera.device_id}: {e}"
                ) from e

        # Connect all cameras concurrently
        await asyncio.gather(
            *[connect_one(name, cam) for name, cam in self.cameras.items()]
        )
        self._connected = True

    def read_frames(self, timeout: float = 1.0) -> dict[str, Frame] | None:
        """
        Read frames from all cameras synchronously with timeout.

        Args:
            timeout: Maximum seconds to wait for all cameras

        Returns:
            Dictionary mapping camera name to Frame, or None if timeout/error
        """
        if not self._connected:
            print("Warning: Cameras not connected")
            return None

        try:
            # Use physicalai's read_cameras for synchronized multi-camera capture
            from physicalai.capture.multi import read_cameras

            synced = read_cameras(self.cameras, timeout=timeout, latest=True)

            # Convert Frame objects to numpy arrays for backward compatibility
            frames = {}
            for name, frame in synced.frames.items():
                frames[name] = frame.data  # Extract numpy array

            return frames

        except CaptureTimeoutError as e:
            print(f"Warning: Camera read timeout: {e}")
            return None
        except CaptureError as e:
            print(f"Warning: Camera read error: {e}")
            return None

    async def async_read_frames(self, timeout: float = 1.0) -> dict[str, Frame] | None:
        """
        Read frames from all cameras asynchronously.

        Args:
            timeout: Maximum seconds to wait for all cameras

        Returns:
            Dictionary mapping camera name to numpy array, or None if timeout/error
        """
        if not self._connected:
            print("Warning: Cameras not connected")
            return None

        try:
            # Use async_read_cameras for concurrent capture
            synced = await async_read_cameras(
                self.cameras, timeout=timeout, latest=True
            )

            # Convert Frame objects to numpy arrays for backward compatibility
            frames = {}
            for name, frame in synced.frames.items():
                frames[name] = frame.data  # Extract numpy array

            return frames

        except CaptureTimeoutError as e:
            print(f"Warning: Camera read timeout: {e}")
            return None
        except CaptureError as e:
            print(f"Warning: Camera read error: {e}")
            return None

    def close(self) -> None:
        """Release all cameras synchronously."""
        for name, camera in self.cameras.items():
            if camera.is_connected:
                camera.disconnect()
                print(f"✓ Disconnected camera '{name}'")
        self._connected = False

    async def async_close(self) -> None:
        """Release all cameras asynchronously."""
        loop = asyncio.get_running_loop()

        async def disconnect_one(name: str, camera: UVCCamera) -> None:
            if camera.is_connected:
                await loop.run_in_executor(None, camera.disconnect)
                print(f"✓ Disconnected camera '{name}'")

        await asyncio.gather(
            *[disconnect_one(name, cam) for name, cam in self.cameras.items()]
        )
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if all cameras are connected."""
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.async_connect()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.async_close()
