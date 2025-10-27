from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from backend_python.gesture_recognition.motion_detector import MotionDetector

router = APIRouter()
detector = MotionDetector()  # ✅ 초기화 1회만 실행

@router.websocket("/ws/motion")
async def motion_ws(websocket: WebSocket):
    """실시간 제스처 인식 WebSocket 엔드포인트"""
    await websocket.accept()
    print("[INFO] Motion WebSocket Connected ✅")

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_text("[ERROR] Webcam not accessible ❌")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_text("[WARN] Frame read failed ⚠️")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            events = detector.detect(frame_rgb)

            if events:
                for e in events:
                    await websocket.send_json({"event": e})
                    print("[EVENT]", e)

    except WebSocketDisconnect:
        print("[INFO] Motion WebSocket Disconnected ❌")

    finally:
        cap.release()
        cv2.destroyAllWindows()
