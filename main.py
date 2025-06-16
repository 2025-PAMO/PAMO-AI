import cv2
import numpy as np
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from motion_detector import MotionDetector
# from motion_detector_movenet import MoveNetDetector

app = FastAPI()
movenet_detector = MotionDetector()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

            # 분석
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