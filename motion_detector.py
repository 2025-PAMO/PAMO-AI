from __future__ import annotations
import os
import time
from typing import Set
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import deque
from pathlib import Path

# =========================================================
# 설정
# =========================================================
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "pamo_static_7gesture_2.h5"

# 모델 로드
model = load_model(MODEL_PATH)
GESTURES = [
    "PointingDown", "PointingUp", "OpenPalm",
    "ClosedFist", "Victory", "SmallHeart", "BigHeart"
]

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================================================
# 유틸 함수
# =========================================================
def count_fingers(landmarks):
    """손가락 개수 계산"""
    tips = [8, 12, 16, 20]
    mcp = [5, 9, 13, 17]
    cnt = 0
    for t, m in zip(tips, mcp):
        if landmarks[t].y < landmarks[m].y:
            cnt += 1
    return cnt


# =========================================================
# MotionDetector 클래스
# =========================================================
class MotionDetector:
    def __init__(self):
        self.prev_label = None
        self.cooldown = 0
        self.cooldowns = {}
        self.frame_idx = 0
        self.last_emit_time = 0
        self.recent_preds = deque(maxlen=5)

        # 쓰로틀/쿨다운 설정
        self.THROTTLE_SEC = 1.0
        self.COOLDOWN_FRAMES = 30

        # 이벤트 매핑
        self.DJ_MAP = {
            "PointingDown": ("pitch_down", "포인팅 다운"),
            "PointingUp": ("pitch_up", "포인팅 업"),
            "OpenPalm": ("speed_up", "손바닥 펴기"),
            "ClosedFist": ("speed_down", "주먹 쥐기"),
            "Victory": ("reverb_toggle", "브이 포즈"),
            "SmallHeart": ("fx_macro_toggle", "손가락 하트"),
            "BigHeart": ("chord_toggle", "두손 하트"),
        }

    # ----------------------------------------
    # 내부 유틸
    # ----------------------------------------
    def _can_emit(self):
        now = time.monotonic()
        if now - self.last_emit_time >= self.THROTTLE_SEC:
            self.last_emit_time = now
            return True
        return False

    # ----------------------------------------
    # 메인 함수
    # ----------------------------------------
    def detect(self, frame_rgb: np.ndarray) -> Set[str]:
        """프레임 입력 → 이벤트 세트 반환"""
        events = set()
        current_label = None
        finger_count = 0

        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            return events

        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            all_joints = []
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4), dtype=np.float32)
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, 1.0]  
                all_joints.append(joint.flatten())

                # 손가락 개수 계산 (한 손 기준)
                if not finger_count:
                    finger_count = count_fingers(res.landmark)

            # 좌표 병합
            if len(all_joints) == 2:
                sample = np.concatenate([all_joints[0], all_joints[1]])
            elif len(all_joints) == 1:
                zeros = np.zeros(21 * 4, dtype=np.float32)
                sample = sample.reshape(1, -1).astype(np.float32, copy=False)       
            else:
                sample = None

            if sample is not None:
                sample = sample.reshape(1, -1)
                probs = model.predict(sample, verbose=0)[0]
                sorted_probs = np.sort(probs)
                top1, top2 = sorted_probs[-1], sorted_probs[-2]
                label = int(np.argmax(probs))

                # 확신도 및 margin 필터
                if top1 > 0.9 and (top1 - top2) > 0.25:
                    self.recent_preds.append(label)

                # 스무딩 (5프레임 중 3프레임 이상 동일)
                if len(self.recent_preds) == 5 and self.recent_preds.count(label) >= 3:
                    current_label = GESTURES[label]
                else:
                    current_label = None

                # 손가락 개수 기반 보정
                if current_label == "Victory" and finger_count != 2:
                    current_label = None
                elif current_label == "OpenPalm" and finger_count < 4:
                    current_label = None
                elif current_label == "ClosedFist" and finger_count > 1:
                    current_label = None

        # 쿨다운 및 이벤트 발생
        if self.cooldown > 0:
            self.cooldown -= 1
        elif current_label and self._can_emit() and current_label != self.prev_label:
            if current_label in self.DJ_MAP:
                key, desc = self.DJ_MAP[current_label]
                events.add(f"{key}, {desc}")
                self.prev_label = current_label
                self.cooldown = self.COOLDOWN_FRAMES

        return events


# =========================================================
# 테스트 실행
# =========================================================
if __name__ == "__main__":
    print("[INFO] MotionDetector 커스텀 모델 전용 버전 ✅")
    cap = cv2.VideoCapture(0)
    detector = MotionDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        events = detector.detect(rgb)

        for e in events:
            print("[EVENT]", e)

        cv2.imshow("MotionDetector (Custom)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
