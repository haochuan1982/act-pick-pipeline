import cv2
import numpy as np
import time
from pathlib import Path

try:
    from openvino import Core  # OpenVINO 2024+
except ImportError:
    from openvino.runtime import Core  # OpenVINO 2023

class ACTModelInference:
    """OpenVINO ACT model inference"""

    def __init__(self, model_path, device='CPU'):
        self.model_path = Path(model_path)

        # Load OpenVINO model
        ie = Core()
        model = ie.read_model(model=str(self.model_path / 'act.xml'))
        self.compiled_model = ie.compile_model(model=model, device_name=device)

        # Get input/output ports
        self.input_ports = {inp.any_name: inp for inp in self.compiled_model.inputs}
        self.output_port = self.compiled_model.output(0)

        print(f"✓ Loaded OpenVINO model from {self.model_path}")
        print(f"  Input shapes:")
        for name, port in self.input_ports.items():
            print(f"    {name}: {port.shape}")
        print(f"  Output shape: {self.output_port.shape}")

        self.target_size = 384  # Model expects 384x384 images

    def preprocess_image(self, image):
        """
        Preprocess image: resize to 384x384 with letterbox, normalize to [0,1], CHW format
        """
        # Letterbox resize to maintain aspect ratio
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # Convert BGR to RGB and normalize to [0, 1]
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        normalized = canvas_rgb.astype(np.float32) / 255.0

        # HWC to CHW format
        chw = np.transpose(normalized, (2, 0, 1))

        return chw

    def infer(self, state, images_dict):
        t_start = time.time()

        # Validate inputs
        if state is None:
            raise ValueError("state cannot be None")
        required_keys = ['top', 'front', 'head']
        missing_keys = [k for k in required_keys if k not in images_dict]
        if missing_keys:
            raise ValueError(f"Missing required camera images: {missing_keys}")

        # Build inputs dict - dynamically match port names to image keys
        t0 = time.time()
        inputs = {}

        # Add state input
        for port_name in self.input_ports.keys():
            if 'state' in port_name.lower():
                inputs[port_name] = state.reshape(1, 6).astype(np.float32)
                break

        # Add image inputs - map model port names to our camera names
        for port_name in self.input_ports.keys():
            port_lower = port_name.lower()
            if 'top' in port_lower:
                inputs[port_name] = self.preprocess_image(images_dict['top']).reshape(1, 3, 384, 384)
            elif 'front' in port_lower:
                inputs[port_name] = self.preprocess_image(images_dict['front']).reshape(1, 3, 384, 384)
            elif 'head' in port_lower or 'wrist' in port_lower:
                inputs[port_name] = self.preprocess_image(images_dict['head']).reshape(1, 3, 384, 384)

        t_preprocess = (time.time() - t0) * 1000  # ms

        # Run inference
        t0 = time.time()
        result = self.compiled_model(inputs)
        actions = result[self.output_port]
        t_model = (time.time() - t0) * 1000  # ms

        t_total = (time.time() - t_start) * 1000  # ms
        print(f"    [Inference breakdown] preprocess={t_preprocess:.1f}ms, model={t_model:.1f}ms, total={t_total:.1f}ms")

        # Shape should be (1, 100, 6), squeeze batch dimension
        return actions.squeeze(0)
