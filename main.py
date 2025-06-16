from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from basicmusic_generate.generator import generate_music_file
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
