from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import write as wav_write
import os

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

def generate_music_from_file(file_path: str) -> str:
    waveform, sr = torchaudio.load(file_path)

    # 샘플링 레이트 32000으로 맞춤
    target_sr = 32000
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    # (T,) 형태일 경우 (1, T)로 변경
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # 입력 생성
    inputs = processor(audio=waveform, sampling_rate=target_sr, return_tensors="pt")
    audio_values = model.generate(**inputs, max_new_tokens=1024)

    # float32 → int16 변환
    output_audio = audio_values[0].cpu().numpy()
    if output_audio.ndim == 1:
        output_audio = np.expand_dims(output_audio, axis=0)
    output_audio = np.clip(output_audio, -1.0, 1.0)
    int16_audio = (output_audio * 32767).astype(np.int16)

    # 저장
    os.makedirs("output", exist_ok=True)
    output_path = "output/generated.wav"
    wav_write(output_path, target_sr, int16_audio.T)
    return output_path
