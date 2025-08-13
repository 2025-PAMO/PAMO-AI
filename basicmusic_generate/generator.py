# generator.py
import io
import logging
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from fastapi import UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SR = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("🎵 MusicGen 모델 로드 중...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
model.eval()
logger.info("✅ 모델 로드 완료 (device=%s)", DEVICE)

async def generate_wav(prompt: str, file: UploadFile | None) -> io.BytesIO:
    """
    프롬프트(+선택적 허밍)를 받아 WAV(BytesIO) 생성 후 반환.
    반환 버퍼는 head로 이동(seek(0))된 상태.
    """
    # 1) 허밍 전처리 (있을 때만)
    melody_wave = None
    if file is not None:
        raw = await file.read()
        if raw:
            seg = AudioSegment.from_file(io.BytesIO(raw))
            seg = seg.set_channels(1).set_frame_rate(SR)
            samples = np.array(seg.get_array_of_samples()).astype(np.float32) / (2 ** 15)
            melody_wave = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).contiguous()

    # 2) 입력 구성
    if melody_wave is not None:
        waveform = melody_wave.squeeze().numpy()
        if waveform.ndim == 2 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        inputs = processor(text=[prompt], audio=waveform, sampling_rate=SR, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], return_tensors="pt")

    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 3) 생성
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=1.5,
            max_new_tokens=512,
        )

    audio = output[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # 4) 원본 WAV를 메모리 버퍼로
    raw_buf = io.BytesIO()
    torchaudio.save(raw_buf, audio.cpu(), sample_rate=SR, format="wav")
    raw_buf.seek(0)

    # 5) (선택) 루프/크로스페이드 처리 — 메모리 내에서만
    seg = AudioSegment.from_file(raw_buf, format="wav")
    duration_ms = len(seg)
    if duration_ms > 5000:
        midpoint = duration_ms // 2
        loop_segment = seg[midpoint:]
        looped = loop_segment
        for _ in range(5):  # 총 6번 반복
            looped = looped.append(loop_segment, crossfade=100)
    else:
        looped = seg

    out_buf = io.BytesIO()
    looped.export(out_buf, format="wav")
    out_buf.seek(0)  # ✅ BytesIO에만 seek

    return out_buf
