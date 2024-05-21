import sys
from pathlib import Path
# import os

# os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

demucs_path = Path(__file__).parent / "demucs"
sys.path.insert(0, str(demucs_path))

from demucs import api as demucs_api

separator = demucs_api.Separator()

origin, separated = separator.separate_audio_file("kissoflife.mp3")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for source_name, audio_tensor in separated.items():
    output_path = output_dir / f"{source_name}_kissoflife.mp3"
    demucs_api.save_audio(audio_tensor, output_path, samplerate=separator.samplerate)
