from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os

# ✅ FastAPI 라우트가 아닌, BytesIO 반환 헬퍼만 import
from basicmusic_generate.generator import generate_wav  # <-- 여기!

app = FastAPI(title="PAMO Backend Python")

# CORS (운영환경에 맞춰 제한 권장)
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
        buf = await generate_wav(prompt=prompt, file=file)  # ✅ BytesIO
        headers = {"Content-Disposition": "attachment; filename=generated_music.wav"}
        return StreamingResponse(buf, media_type="audio/wav", headers=headers)
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
