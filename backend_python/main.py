import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os
from fastapi import WebSocket, WebSocketDisconnect
import cv2, numpy as np, asyncio
from motion_detector import MotionDetector
import json, requests

from basicmusic_generate.generator import generate_music_file 

app = FastAPI(title="PAMO Backend Python")

RESEND_MS = 200   # motions가 같아도 0.2초마다 한 번은 보내기
last_sent_ts = 0

# CORS (운영환경에 맞게 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# 1) 음악 생성 — 허밍 파일은 선택
@app.post("/generate-music")
async def generate_music(
    prompt: str = Form(...),
    file: UploadFile | None = File(None)
):
    try:
        output_path = await generate_music_file(file, prompt)
        return FileResponse(output_path, media_type="audio/wav", filename="generated_music.wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- 썸네일 생성( S3 → 추출 → S3 업로드 ) ----------
class ThumbReq(BaseModel):
    s3_bucket: str
    s3_key: str
    timestamp_sec: float | None = 8.0
    out_s3_bucket: str
    out_s3_prefix: str  # e.g. "motion/thumbnails/"

def _extract_frame_ffmpeg(video_path: str, out_jpg: str, ts: float) -> None:
    cmd = [
        "ffmpeg",
        "-ss", str(ts),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        out_jpg,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@app.post("/generate-thumbnail-from-s3")
def generate_thumbnail_from_s3(req: ThumbReq):
    import boto3
    from botocore.client import Config

    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    prefix = req.out_s3_prefix.strip("/")
    out_key_dir = f"{prefix}/" if prefix else ""

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.mp4")
        out_path = os.path.join(td, "thumb.jpg")

        s3.download_file(req.s3_bucket, req.s3_key, in_path)
        _extract_frame_ffmpeg(in_path, out_path, req.timestamp_sec or 8.0)

        base = os.path.basename(req.s3_key)
        name, _ = os.path.splitext(base)
        out_key = f"{out_key_dir}{name}_thumb.jpg"

        s3.upload_file(out_path, req.out_s3_bucket, out_key, ExtraArgs={"ContentType": "image/jpeg"})
        url = f"https://{req.out_s3_bucket}.s3.amazonaws.com/{out_key}"
        return {
            "thumbnail_bucket": req.out_s3_bucket,
            "thumbnail_key": out_key,
            "thumbnail_url": url,
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    detector = MotionDetector()
    try:
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        detector.detect(dummy)
    except Exception:
        pass

    prev_motions: set[str] = set()
    frames_rx = 0
    last_log = time.time()

    frame_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)

    async def receiver():
        nonlocal frames_rx, last_log
        try:
            while True:
                msg = await websocket.receive()
                typ = msg.get("type")

                if typ == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    np_arr = np.frombuffer(msg["bytes"], np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if frame_q.full():
                        try:
                            frame_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_q.put(frame)

                    frames_rx += 1
                    now = time.time()
                    if now - last_log >= 3:
                        print(f"[WS] frames rx ≈ {frames_rx} (last 3s)")
                        frames_rx = 0
                        last_log = now
                    continue

                if "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        data = {"type": msg["text"]}

                    t = (data.get("type") or "").lower()
                    if t == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif t == "auth":
                        await websocket.send_json({"type": "auth_ok"})
                    continue
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print("❌ receiver error:", repr(e))

    last_sent_ts_ms = 0

    async def processor():
        nonlocal prev_motions, last_sent_ts_ms
        try:
            while True:
                frame_bgr = await frame_q.get()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                motions = safe_detect(detector, frame_rgb)
                now_ms = int(time.time() * 1000)
                changed = motions != prev_motions
                time_ok = (now_ms - last_sent_ts_ms) >= RESEND_MS

                if changed or (motions and time_ok):
                    await websocket.send_json({
                        "type": "motion",
                        "motions": [{"id": m, "label": m.split(',')[0] if m else ""} for m in motions]
                    })
                    prev_motions = set(motions)
                    last_sent_ts_ms = now_ms
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print("❌ processor error:", repr(e))

    recv_task = asyncio.create_task(receiver())
    proc_task = asyncio.create_task(processor())

    try:
        await asyncio.gather(recv_task, proc_task)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def safe_detect(detector: MotionDetector | None, frame_rgb) -> set[str]:
    if detector is None:
        return set()
    try:
        out = detector.detect(frame_rgb)
        if not out:
            return set()
        if isinstance(out, (list, tuple, set)):
            return {str(x) for x in out}
        return {str(out)}
    except Exception as e:
        print("⚠️ detect() error suppressed:", repr(e))
        return set()

async def process_frame_and_reply(websocket, frame_bgr, prev_motions, detector):
    global last_sent_ts
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)

        motions = safe_detect(detector, frame_rgb)   
        if motions:
            print("motions:", motions)
        else:
            # print("motions: ∅") # 로그 너무 많이 생성되는 것 같으면 주석 처리해도 될 것 같음.
            pass

        now = time.time() * 1000
        changed = motions != prev_motions
        time_ok = (now - last_sent_ts) >= RESEND_MS

        if changed or (motions and time_ok):
            payload = {
                "type": "motion",
                "motions": [{"id": m, "label": (m.split(',')[0] if m else "")}
                            for m in sorted(motions)]
            }
            await websocket.send_json(payload)
            prev_motions.clear()
            prev_motions.update(motions)
            last_sent_ts = now

    except WebSocketDisconnect:
        return
    except Exception as e:
        print("❌ process_frame_and_reply error:", repr(e))

@app.get("/proxy-audio")
def proxy_audio(url: str):
    try:
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "audio/wav")
        return StreamingResponse(r.iter_content(chunk_size=8192), media_type=content_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})