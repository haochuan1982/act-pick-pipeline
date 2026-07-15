from physicalai.capture import SharedCamera
from physicalai.capture import CameraType, ColorMode

# Try RealSense camera which is more reliable
with SharedCamera(camera_type=CameraType.UVC, camera_kwargs={"device": "/dev/video4", 'width': 640, 'height': 480, 'fps': 30}) as camera:
    frame = camera.read_latest()
    print(frame.data.shape)  # (480, 640, 3)
    print(frame.timestamp)   # monotonic timestamp


