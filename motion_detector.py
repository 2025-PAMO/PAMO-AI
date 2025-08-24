# motion_detector.py
# - 프론트 연동 그대로: MotionDetector.detect(frame_rgb) -> set[str]
# - 내부는 MediaPipe Tasks GestureRecognizer(7제스처) 사용
# - 전역 1초 쓰로틀 + 이벤트별 쿨다운 유지
# - 반환 문자열 포맷: "event_key, desc" (기존과 동일)

from __future__ import annotations
import time
import os
from typing import Set, Optional

import numpy as np
import mediapipe as mp

# ===== 설정 기본값 =====
# ⚠️ 항상 motion_detector.py 파일이 있는 디렉토리 기준으로 .task 파일 찾음
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
DEFAULT_CONF_THRESH = 0.65
DEFAULT_COOLDOWN_FRAMES = 18          # ≈0.6s@30fps
DEFAULT_FPS_ESTIMATE = 30
DEFAULT_THROTTLE_SECONDS = 1.0        # 1초에 1건만 이벤트 허용
DEFAULT_IGNORE_LABELS = {"None"}      # 노이즈 라벨 무시

# MediaPipe Tasks 단축
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MotionDetector:
    """
    프론트 호환 클래스:
      - __init__(...) 파라미터로 임계/쿨다운/모델 경로 설정 가능
      - detect(frame_rgb: np.ndarray) -> Set[str]
        * 이벤트가 있으면 {"event_key, desc"} 한 개 반환 (기존 규약과 동일)
        * 없으면 빈 set()
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_thresh: float = DEFAULT_CONF_THRESH,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
        fps_estimate: int = DEFAULT_FPS_ESTIMATE,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        ignore_labels: Optional[Set[str]] = None,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.cooldown_frames = cooldown_frames
        self.fps_estimate = max(int(fps_estimate), 1)
        self.throttle_seconds = float(throttle_seconds)
        self.ignore_labels = set(ignore_labels) if ignore_labels is not None else set(DEFAULT_IGNORE_LABELS)

        # GestureRecognizer 초기화
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        # 상태값
        self.frame_idx = 0
        self.cooldowns = {}           # event_key -> 남은 프레임 수
        self.last_event_key = None    # 연속 동일 이벤트 방지
        self.last_emit_time = 0.0     # 전역 쓰로틀(초)

        # 라벨 → (이벤트키, 설명) 매핑 (7제스처 균형 배치)
        self.DJ_MAP = {
            "Open_Palm":   ("speed_up",        "+5%"),
            "Closed_Fist": ("speed_down",      "-5%"),
            "Pointing_Up": ("tap_tempo",       "tap"),
            "Thumb_Up":    ("pitch_up",        "+1st"),
            "Thumb_Down":  ("pitch_down",      "-1st"),
            "Victory":     ("chord_toggle",    "latch"),
            "ILoveYou":    ("fx_macro_toggle", "toggle"),
        }

    # ---------- 내부 유틸 ----------
    def _timestamp_ms(self) -> int:
        ts = int((self.frame_idx / self.fps_estimate) * 1000)
        self.frame_idx += 1
        return ts

    def _can_emit_now(self) -> bool:
        now = time.monotonic()
        if (now - self.last_emit_time) >= self.throttle_seconds:
            self.last_emit_time = now
            return True
        return False

    def _cooling(self, key: str) -> bool:
        if key in self.cooldowns and self.cooldowns[key] > 0:
            self.cooldowns[key] -= 1
            return True
        return False

    def _decay_cooldowns(self):
        for k in list(self.cooldowns.keys()):
            if self.cooldowns[k] > 0:
                self.cooldowns[k] -= 1
            else:
                del self.cooldowns[k]

    # ---------- 외부 인터페이스 ----------
    def detect(self, frame_rgb: np.ndarray) -> Set[str]:
        events: Set[str] = set()

        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            self._decay_cooldowns()
            return events

        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.recognizer.recognize_for_video(mp_image, self._timestamp_ms())

        if result.gestures:
            top = result.gestures[0][0]
            label = top.category_name
            score = top.score

            if label not in self.ignore_labels and score >= self.conf_thresh:
                if label in self.DJ_MAP:
                    event_key, desc = self.DJ_MAP[label]
                    if self._can_emit_now() and not self._cooling(event_key) and event_key != self.last_event_key:
                        self.cooldowns[event_key] = self.cooldown_frames
                        self.last_event_key = event_key
                        events.add(f"{event_key}, {desc}")
                        self._decay_cooldowns()
                        return events

        self._decay_cooldowns()
        return events


if __name__ == "__main__":
    print("[INFO] MotionDetector class is ready. Import this in your frontend loop.")
