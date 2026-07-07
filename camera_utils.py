from __future__ import annotations
from typing import Any

from physicalai.capture import CameraType, ColorMode, SharedCamera

_DRIVER_TO_CAMERA_TYPE: dict[str, CameraType] = {
    "usb_camera": CameraType.UVC,
    "realsense": CameraType.REALSENSE,
    "basler": CameraType.BASLER,
}

def driver_to_camera_type(driver: str) -> CameraType:
    try:
        return _DRIVER_TO_CAMERA_TYPE[driver]
    except KeyError:
        msg = f"unsupported driver {driver!r};"
        raise ValueError(msg) from None

def _camera_type_and_kwargs(config: dict[str, Any]) -> tuple[CameraType, dict[str, Any]]:
    camera_type = driver_to_camera_type(config["driver"])

    payload = config["payload"]
    camera_kwargs: dict[str, Any] = {k: v for k, v in payload.items() if v is not None}

    if camera_type == CameraType.UVC:
        camera_kwargs["device"] = config["fingerprint"]
    else:
        camera_kwargs["serial_number"] = config["serial_number"]

    return camera_type, camera_kwargs

def build_shared_camera(
    config: dict[str, Any],
    *,
    validate_on_connect: bool = False,
    overwrite_settings: bool = False,
    idle_timeout: float = 5.0,
) -> SharedCamera:
    camera_type, camera_kwargs = _camera_type_and_kwargs(config)

    return SharedCamera(
        camera_type,
        color_mode=ColorMode.RGB,
        validate_on_connect=validate_on_connect,
        overwrite_settings=overwrite_settings,
        idle_timeout=idle_timeout,
        **camera_kwargs,
    )


