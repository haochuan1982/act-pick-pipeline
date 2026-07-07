#!/usr/bin/env python3
"""Test script for CameraManager with physicalai."""

from CameraManager import CameraManager

def test_camera_manager():
    """Test basic CameraManager functionality."""

    # Test with a simple configuration
    # Note: Adjust device IDs based on your actual setup
    CAMERA_MAPPING = {
        'top': 8,     # UGREEN Camera 1080P
        'front': 6,   # USB2.0_CAM1
        'head': 2,    # RealSense 435i RGB
    }

    print("=" * 60)
    print("Testing CameraManager with physicalai")
    print("=" * 60)

    try:
        # Create camera manager
        print("\n[1/3] Creating CameraManager...")
        cameras = CameraManager(CAMERA_MAPPING)
        print(f"✓ CameraManager created with {len(cameras.cameras)} cameras")

        # Connect to cameras
        print("\n[2/3] Connecting to cameras...")
        cameras.connect(timeout=5.0)
        print(f"✓ All cameras connected: {cameras.is_connected}")

        # Read a few frames
        print("\n[3/3] Reading test frames...")
        for i in range(3):
            frames = cameras.read_frames(timeout=1.0)
            if frames is not None:
                print(f"  Frame {i+1}: ", end="")
                for name, frame in frames.items():
                    print(f"{name}={frame.shape} ", end="")
                print()
            else:
                print(f"  Frame {i+1}: Read failed")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if 'cameras' in locals():
            cameras.close()
        print("✓ Done")

if __name__ == '__main__':
    test_camera_manager()
