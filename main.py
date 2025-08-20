import cv2
import numpy as np
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from motion_detector import MotionDetector
from basicmusic_generate.generator import generate_music_file

# 로깅 설정
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정 통합
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"] 로 조정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 음악 생성 API
@app.post("/generate-music")
async def generate_music(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        output_path = await generate_music_file(file, prompt)
        return FileResponse(output_path, media_type="audio/wav", filename="generated_music.wav")
    except Exception as e:
        logger.error("API 처리 실패: %s", e)
        return {"error": "음악 생성 실패"}

# ✅ 모션 인식 WebSocket
@app.websocket("/motion-detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    detector = MotionDetector()
    prev_motions = set()

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError as e:
                print(f"❌ receive() 실패: {e}")
                break

            if "bytes" not in message:
                continue

            img_bytes = message["bytes"]
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            motions = detector.detect(frame_rgb)

            if motions != prev_motions:
                await websocket.send_text(json.dumps({
                    "type": "motion",
                    "motions": [
                        {"id": m, "label": m.replace('_', ' ').title()} for m in motions
                    ]
                }, ensure_ascii=False))
                prev_motions = motions.copy()

            await asyncio.sleep(0.03)  # 약 30fps 대응

    except WebSocketDisconnect:
        print("🔌 클라이언트 연결 해제")
