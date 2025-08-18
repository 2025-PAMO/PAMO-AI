# pitch_natural_prompt.py

import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging

app = FastAPI()
logger = logging.getLogger("pitch_natural_prompt")
logging.basicConfig(level=logging.INFO)

# 음높이 → 음이름 (librosa 기본 MIDI→note)
def midi_to_note(midi_num):
    return librosa.midi_to_note(midi_num)

# 음성 파일에서 피치 + 지속시간 추출
def extract_pitch_and_duration(y, sr, hop_length=512):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)

    notes = []
    prev_note = None
    duration = 0
    time_per_frame = hop_length / sr

    for i in range(pitches.shape[1]):
        pitch_slice = pitches[:, i]
        mag_slice = magnitudes[:, i]
        if mag_slice.any():
            idx = mag_slice.argmax()
            pitch = pitch_slice[idx]
            if pitch > 0:
                midi = int(librosa.hz_to_midi(pitch))
                note = midi_to_note(midi)

                if note == prev_note:
                    duration += time_per_frame
                else:
                    if prev_note:
                        notes.append((prev_note, round(duration, 2)))
                    prev_note = note
                    duration = time_per_frame
            else:
                if prev_note:
                    notes.append((prev_note, round(duration, 2)))
                    prev_note = None
                    duration = 0
        else:
            if prev_note:
                notes.append((prev_note, round(duration, 2)))
                prev_note = None
                duration = 0

    if prev_note:
        notes.append((prev_note, round(duration, 2)))
    return notes

# 자연어 문장으로 변환
def to_natural_language(notes, user_prompt=None):
    note_descriptions = [f"{note} ({duration}s)" for note, duration in notes]
    melody_description = ", ".join(note_descriptions)

    if user_prompt:
        return f"{user_prompt}. The melody includes notes like {melody_description}."
    else:
        return f"A melody composed of {melody_description}."

# ✅ API 엔드포인트
@app.post("/generate-natural-prompt")
async def generate_natural_prompt(file: UploadFile = File(...), prompt: str = Form(None)):
    logger.info(f"📩 파일 수신: {file.filename} | 사용자 프롬프트: {prompt}")

    y, sr = librosa.load(file.file, sr=None)
    notes = extract_pitch_and_duration(y, sr)

    natural_prompt = to_natural_language(notes, prompt)
    logger.info(f"📝 생성된 프롬프트: {natural_prompt}")

    return JSONResponse(content={"natural_prompt": natural_prompt})


# 로컬 실행용
if __name__ == "__main__":
    uvicorn.run("pitch_natural_prompt:app", host="127.0.0.1", port=8020, reload=True)
