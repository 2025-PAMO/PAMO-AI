from __future__ import annotations
import time
import os
from typing import Set
import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2

# ===== 설정 =====
BASE_DIR = os.path.dirname(__file__)

# FastAPI 배포용 — 고정 경로 (최종 모델만 로드)
CUSTOM_MODEL_PATH = os.path.join(
    BASE_DIR, "gesture-recognition", "models", "final", "pamo_static_7gesture_2.h5"
)

DEFAULT_CONF_THRESH = 0.65   # MediaPipe 기본 제스처 임계값
HEART_CONF_THRESH = 0.85     # 하트 모델은 더 빡세게
HEART_MARGIN_THRESH = 0.2    # 하트 1등-2등 확률 차이 최소
DEFAULT_COOLDOWN_FRAMES = 18
DEFAULT_FPS_ESTIMATE = 30
DEFAULT_THROTTLE_SECONDS = 1.0

# MediaPipe 단축
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MotionDetector:
    def __init__(self):
        # MediaPipe 기본 제스처 인식기
        options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.join(BASE_DIR, "gesture_recognizer.task")
            ),
            running_mode=VisionRunningMode.VIDEO,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        # 커스텀 모델 (최종)
        self.heart_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
        self.heart_actions = ["FingerHeart", "TwoHandHeart", "WingHeart"]

        # MediaPipe Hands (커스텀 모델 feature 추출용)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # 상태값 초기화
        self.frame_idx = 0
        self.cooldowns = {}
        self.last_event_key = None
        self.last_emit_time = 0.0
        self.heart_streak = {"label": None, "count": 0}

        # ✅ 프론트와 연동되는 이벤트 매핑
        self.DJ_MAP = {
            "Thumb_Down": ("pitch_down", "포인팅 다운"),
            "Thumb_Up": ("pitch_up", "포인팅 업"),
            "Open_Palm": ("speed_up", "손바닥 펴기"),
            "Closed_Fist": ("speed_down", "주먹 쥐기"),
            "Victory": ("reverb_toggle", "브이 포즈"),
            "ILoveYou": ("fx_macro_toggle", "손가락 하트"),
            "Heart": ("chord_toggle", "두손 하트"),  # BigHeart 대체
        }

    # ---------- 내부 유틸 ----------
    def _timestamp_ms(self) -> int:
        ts = int((self.frame_idx / DEFAULT_FPS_ESTIMATE) * 1000)
        self.frame_idx += 1
        return ts

    def _can_emit_now(self) -> bool:
        now = time.monotonic()
        if (now - self.last_emit_time) >= DEFAULT_THROTTLE_SECONDS:
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

    # ---------- 메인 ----------
    def detect(self, frame_rgb: np.ndarray) -> Set[str]:
        events: Set[str] = set()
        current_event_key, desc = None, None

        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            self._decay_cooldowns()
            return events

        # 1️⃣ 기본 MediaPipe 제스처 인식
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.recognizer.recognize_for_video(mp_image, self._timestamp_ms())
        if result.gestures:
            top = result.gestures[0][0]
            if top.category_name in self.DJ_MAP and top.score >= DEFAULT_CONF_THRESH:
                current_event_key, desc = self.DJ_MAP[top.category_name]

        # 2️⃣ fallback (커스텀 모델)
        if current_event_key is None:
            hand_result = self.hands.process(frame_rgb)
            all_joints = []
            if hand_result.multi_hand_landmarks:
                for res in hand_result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    all_joints.append(joint.flatten())

            if all_joints:
                if len(all_joints) == 2:
                    sample = np.concatenate([all_joints[0], all_joints[1]])
                else:
                    zeros = np.zeros(21 * 4)
                    sample = np.concatenate([all_joints[0], zeros])

                sample = sample.reshape(1, -1)
                pred = self.heart_model.predict(sample, verbose=0)[0]
                class_id = int(np.argmax(pred))
                confidence = pred[class_id]

                if confidence >= HEART_CONF_THRESH:
                    label = self.heart_actions[class_id]
                    if self.heart_streak["label"] == label:
                        self.heart_streak["count"] += 1
                    else:
                        self.heart_streak = {"label": label, "count": 1}

                    if self.heart_streak["count"] >= 2 and label in self.DJ_MAP:
                        current_event_key, desc = self.DJ_MAP[label]

        # 3️⃣ 이벤트 발생
        if current_event_key and self._can_emit_now() and not self._cooling(current_event_key):
            self.cooldowns[current_event_key] = DEFAULT_COOLDOWN_FRAMES
            self.last_event_key = current_event_key
            events.add(f"{current_event_key}, {desc}")

        self._decay_cooldowns()
        return events


# ==== 테스트용 ====
if __name__ == "__main__":
    print("[INFO] MotionDetector ready ✅")

    cap = cv2.VideoCapture(0)
    detector = MotionDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        events = detector.detect(frame_rgb)
        for e in events:
            print("[EVENT]", e)

        cv2.imshow("MotionDetector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
