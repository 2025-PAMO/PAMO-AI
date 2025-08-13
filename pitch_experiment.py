from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import librosa
import torchaudio
import torch
import tempfile
import os
import io
import numpy as np
import logging

# ── 로깅 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pitch_experiment")

# ── 앱 & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(title="MusicGen (Pitch-only)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요시 특정 오리진으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 모델 로드 ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("🎵 MusicGen 모델 로드 중...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
model.eval()
logger.info("✅ 모델 로드 완료 (device=%s)", DEVICE)

# ── Pitch 추출 ───────────────────────────────────────────────────────────────
def extract_pitches(file_path: str, max_notes: int = 30) -> list[str]:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    seq = []
    for frame in pitches.T:
        idx = int(np.argmax(frame))
        pitch = float(frame[idx])
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            seq.append(note)
            if len(seq) >= max_notes:
                break
    return seq

# ── 유틸: 텐서를 WAV 바이트로 ───────────────────────────────────────────────
def _to_wav_bytes(audio_tensor: torch.Tensor) -> io.BytesIO:
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.cpu(), sample_rate=SAMPLE_RATE, format="wav")
    buffer.seek(0)
    return buffer

# ── 피치 기반 프롬프트 + 텍스트 프롬프트 ────────────────────────────────────
@app.post("/generate-music-pitch-only")
async def generate_music_with_pitch(file: UploadFile = File(...), prompt: str = Form(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file이 필요합니다.")
    logger.info("📩 [pitch-only] 프롬프트='%s' | 파일=%s", prompt, getattr(file, "filename", ""))

    # 업로드를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 피치 → 음표 시퀀스
        pitch_tokens = extract_pitches(tmp_path)
        pitch_str = " ".join(pitch_tokens) if pitch_tokens else "C4"

        # 최종 프롬프트
        full_prompt = f"melody: {pitch_str}. style: {prompt}"
        logger.info("🎯 최종 프롬프트: %s", full_prompt)

        # MusicGen 추론
        inputs = processor(text=[full_prompt], return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        logger.info("🧠 모델 추론 시작 (pitch+text)")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=1.5,
                max_new_tokens=512,
            )

        buffer = _to_wav_bytes(output[0].detach())
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        logger.exception("피치 기반 생성 실패: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ── 호환용 엔드포인트: /generate-music → A안 매핑 ──────────────────────────
@app.post("/generate-music")
async def generate_music_compat(file: UploadFile = File(...), prompt: str = Form(...)):
    return await generate_music_with_pitch(file=file, prompt=prompt)

# ── 로컬 실행 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pitch_experiment:app", host="0.0.0.0", port=8010, reload=True)
