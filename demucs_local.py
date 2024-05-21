import sys
from pathlib import Path
import torch as th
# import os

# os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

demucs_path = Path(__file__).parent / "demucs"
sys.path.insert(0, str(demucs_path))

from demucs import api as demucs_api
from demucs import utils

separator = demucs_api.Separator()

padded_mix, model, length = separator.get_padded_mix("kissoflife.mp3")

out = model(padded_mix)
assert isinstance(out, th.Tensor)
final_out = utils.center_trim(out, length)
print(final_out)
# print(htdemucs_model)

# origin, separated = separator.separate_audio_file("kissoflife.mp3")

# print(type(separated))
# print(separated)

# output_dir = Path("output")
# output_dir.mkdir(exist_ok=True)

# for source_name, audio_tensor in separated.items():
#     output_path = output_dir / f"{source_name}_kissoflife.mp3"
#     demucs_api.save_audio(audio_tensor, output_path, samplerate=separator.samplerate)
