from basicmusic_generate.generator import generate_music_file
import torchaudio
import torch
import os
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment
import tempfile
import logging

logger = logging.getLogger(__name__)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def generate_music_file(file, prompt: str) -> str:
    sr = 32000
    melody_tensor = None

    try:
        if file:
            logger.info("🔄 Step 1: 허밍 전처리 시작")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_channels(1).set_frame_rate(sr)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)
            melody_tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).contiguous()
            os.remove(tmp_path)

        logger.info("🧠 Step 2: 모델 입력 구성 시작")
        if melody_tensor is not None:
            waveform = melody_tensor.squeeze().numpy()
            if waveform.ndim == 2 and waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            inputs = processor(
                text=[prompt],
                audio=waveform,
                sampling_rate=sr,
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=[prompt],
                return_tensors="pt"
            )

        logger.info("🎼 Step 3: 모델 추론 시작")
        output = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=1.5,
            max_new_tokens=512
        )

        output_tensor = output[0]
        if output_tensor.dim() == 1:
            output_tensor = output_tensor.unsqueeze(0)

        raw_path = os.path.join(OUTPUT_DIR, "generated_raw.wav")
        torchaudio.save(raw_path, output_tensor, sample_rate=sr)
        logger.info(f"💾 원본 저장 완료: {raw_path}")

        # 루프 생성
        segment = AudioSegment.from_wav(raw_path)
        duration_ms = len(segment)
        midpoint = duration_ms // 2

        if duration_ms > 5000:
            loop_segment = segment[midpoint:]
            looped = loop_segment
            for _ in range(5):
                looped = looped.append(loop_segment, crossfade=100)
            logger.info("🔁 루프 처리 완료")
        else:
            looped = segment
            logger.warning("⚠️ 루프 없이 반환")

        final_path = os.path.join(OUTPUT_DIR, "generated_music_looped.wav")
        looped.export(final_path, format="wav")
        logger.info(f"✅ 최종 파일 저장 완료: {final_path}")

        return final_path

    except Exception as e:
        logger.exception("❌ 음악 생성 실패: %s", str(e))
        raise RuntimeError("음악 생성 실패")
