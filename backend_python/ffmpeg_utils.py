import json
import subprocess

def _probe_duration_sec(path: str) -> float | None:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "json", path
    ]
    try:
        out = subprocess.check_output(cmd)
        dur = json.loads(out.decode()).get("format", {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None

def extract_thumbnail_ffmpeg(input_path: str, output_path: str, timestamp: float | None):
    ts = 8.0 if (timestamp is None) else float(timestamp)
    dur = _probe_duration_sec(input_path)
    if dur and dur > 0:
        max_ts = max(0.0, dur - 0.1)
        if ts > max_ts: ts = max_ts
        if ts < 0: ts = 0.0

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ss", f"{ts:.3f}",
        "-frames:v", "1",
        "-q:v", "2",
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
