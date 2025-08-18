from fastapi import UploadFile
import torchaudio
import torch
import os
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment
import tempfile
import logging
import librosa

logger = logging.getLogger(__name__)

# 모델 준비
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎼 Pitch 추출 함수
def extract_pitches(file_path):
    y, sr = librosa.load(file_path, sr=32000)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_sequence = []
    for frame in pitches.T:
        idx = frame.argmax()
        pitch = frame[idx]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            pitch_sequence.append(note)
    return pitch_sequence[:30]  # 너무 길어지지 않게 앞부분만 사용

# 🎶 메인 함수
async def generate_music_file(file: UploadFile, prompt: str) -> str:
    sr = 32000
    try:
        # 1) 업로드된 파일을 임시 wav로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2) 피치 추출 → 프롬프트 확장
        pitch_tokens = extract_pitches(tmp_path)
        pitch_str = " ".join(pitch_tokens)
        full_prompt = f"melody: {pitch_str}. style: {prompt}"
        logger.info("🎯 최종 프롬프트: %s", full_prompt)

        # 3) 모델 입력 구성
        inputs = processor(text=[full_prompt], return_tensors="pt")

        # 4) 모델 추론
        logger.info("🧠 모델 추론 시작")
        output = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=1.5,
            max_new_tokens=512
        )

        output_tensor = output[0]
        if output_tensor.dim() == 1:
            output_tensor = output_tensor.unsqueeze(0)

        # 5) 원본 파일로 저장
        raw_path = os.path.join(OUTPUT_DIR, "generated_raw.wav")
        torchaudio.save(raw_path, output_tensor, sample_rate=sr)
        logger.info(f"💾 원본 저장 완료: {raw_path}")

        # 6) 고정 4회 반복
        segment = AudioSegment.from_wav(raw_path)
        REPEAT_COUNT = 4

        looped = segment
        for _ in range(REPEAT_COUNT - 1):
            looped = looped.append(segment, crossfade=100)

        logger.info(f"🔁 전체 {REPEAT_COUNT}회 반복 완료 "
                    f"({len(segment)/1000:.1f}초 × {REPEAT_COUNT} = {len(looped)/1000:.1f}초)")

        # 7) 최종 저장
        final_path = os.path.join(OUTPUT_DIR, "generated_music_looped.wav")
        looped.export(final_path, format="wav")
        logger.info(f"✅ 최종 파일 저장 완료: {final_path}")

        # 임시파일 삭제
        os.remove(tmp_path)

        return final_path

    except Exception as e:
        logger.exception("❌ 음악 생성 실패: %s", str(e))
        raise RuntimeError("음악 생성 실패")
