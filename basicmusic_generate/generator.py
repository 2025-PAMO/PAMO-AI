import io
import os
import tempfile
import logging
from typing import List

import torch
import torchaudio
import librosa
from pydub import AudioSegment
from fastapi import UploadFile
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# -------------------------------------------------
# 로깅 설정
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SR = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("🎵 MusicGen 모델 로드 중...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
model.eval()
logger.info("✅ 모델 로드 완료 (device=%s)", DEVICE)


# -------------------------------------------------
# 유틸: 피치(음높이) 시퀀스 추출
# -------------------------------------------------
def extract_pitches(file_path: str, max_notes: int = 30) -> List[str]:
    """
    오디오 파일에서 피치를 추출해 음표 시퀀스로 반환.
    너무 길어지는 걸 방지하기 위해 max_notes 만큼만 사용.
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    notes: List[str] = []
    for frame in pitches.T:
        idx = int(frame.argmax())
        freq = float(frame[idx])
        if freq > 0:
            notes.append(librosa.hz_to_note(freq))
        if len(notes) >= max_notes:
            break
    return notes


# -------------------------------------------------
# 메인: 업로드 파일(Optional) + 텍스트 프롬프트로 음악 생성
# -------------------------------------------------
async def generate_music_file(file: UploadFile | None, prompt: str, repeat_count: int = 4) -> str:
    """
    업로드된 오디오가 있으면 피치를 추출해 프롬프트 확장,
    없으면 텍스트 프롬프트만 사용.
    MusicGen으로 음악을 생성하고 crossfade 반복을 적용한 wav 파일 경로를 반환.
    """
    try:
        tmp_path = None
        pitch_str = "none"

        # 1) 업로드 파일 처리 (있을 경우만)
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                raw = await file.read()
                tmp.write(raw)
                tmp_path = tmp.name
            logger.info("📥 업로드 파일 저장: %s", tmp_path)

            # 피치 추출
            pitch_tokens = extract_pitches(tmp_path)
            pitch_str = " ".join(pitch_tokens) if pitch_tokens else "none"
            logger.info("🎹 피치 추출 완료: %s", pitch_str)

        # 2) 최종 프롬프트 구성
        full_prompt = f"melody: {pitch_str}. style: {prompt}"
        logger.info("🎯 최종 프롬프트: %s", full_prompt)

        # 3) 모델 입력 구성
        inputs = processor(text=[full_prompt], return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # 4) 모델 추론
        logger.info("🧠 모델 추론 시작")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=1.5,
                max_new_tokens=512,
            )

        # MusicGen은 wave tensor 반환한다고 가정
        audio = output[0]  # [channels, time] or [time]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, time]

        # 5) 원본 저장
        raw_path = os.path.join(OUTPUT_DIR, "generated_raw.wav")
        torchaudio.save(raw_path, audio.cpu(), sample_rate=SR, format="wav")
        logger.info("💾 원본 저장 완료: %s", raw_path)

        # 6) crossfade 반복
        segment = AudioSegment.from_file(raw_path, format="wav")
        looped = segment
        for _ in range(max(1, repeat_count) - 1):
            looped = looped.append(segment, crossfade=100)

        logger.info("🔁 반복 적용: %.1fs × %d = %.1fs",
            len(segment) / 1000.0, max(1, repeat_count), len(looped) / 1000.0
        )

        # 7) 최종 저장
        final_path = os.path.join(OUTPUT_DIR, "generated_music_looped.wav")
        looped.export(final_path, format="wav")
        logger.info("✅ 최종 파일 저장 완료: %s", final_path)

        # 임시파일 정리
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return final_path

    except Exception as e:
        logger.exception("❌ 음악 생성 실패: %s", str(e))
        raise RuntimeError("음악 생성 실패") from e
