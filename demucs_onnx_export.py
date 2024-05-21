import sys
from pathlib import Path

import torch

demucs_path = Path(__file__).parent / "demucs"
sys.path.insert(0, str(demucs_path))

from demucs import api as demucs_api

separator = demucs_api.Separator()
filename = "kissoflife.mp3"
padded_mix, model, length = separator.get_padded_mix("kissoflife.mp3")

# torch.onnx.export(model, padded_mix, "model.onnx")
onnx_program = torch.onnx.dynamo_export(model, padded_mix)