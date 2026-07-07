from typing import Any
#from pathlib import Path

from physicalai.inference import InferenceModel

def load_inference_model(config: dict[str, Any] ) -> InferenceModel:
    return InferenceModel(
        export_dir=config['model_path'],
        policy_name=config['policy_name'],
        backend=config['backend'],
        device=config['device'],
    )