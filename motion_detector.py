import mediapipe as mp
import math
from collections import deque
import numpy as np

class StarPatternDetector:
    def __init__(self):
        self.trail = deque(maxlen=60)

    def add_point(self, x, y):
        self.trail.append((x, y))

    def _angle_between(self, p1, p2, p3):
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        angle = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        return abs((angle + 180) % 360 - 180)

    def _path_length(self):
        length = 0
        for i in range(1, len(self.trail)):
            dx = self.trail[i][0] - self.trail[i-1][0]
            dy = self.trail[i][1] - self.trail[i-1][1]
            length += math.hypot(dx, dy)
        return length

    def detect_star_like_pattern(self):
        if len(self.trail) < 30:
            return False

        if self._path_length() < 0.5:
            return False

        xs, ys = zip(*self.trail)
        cx, cy = np.mean(xs), np.mean(ys)
        radii = [math.hypot(x - cx, y - cy) for x, y in self.trail]
        radius_variation = np.std(radii)
        if not (0.05 < radius_variation < 0.2):
            return False

        sharp_angles = 0
        for i in range(3, len(self.trail) - 3, 3):
            a = self._angle_between(self.trail[i-3], self.trail[i], self.trail[i+3])
            if 35 < a < 85:
                sharp_angles += 1

        return sharp_angles >= 4

class MotionDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.wrist_angle_history = deque(maxlen=5)
        self.wrist_y_history = deque(maxlen=5)
        self.knee_x_diff_history = deque(maxlen=20)
        self.left_ankle_y_history = deque(maxlen=5)
        self.right_ankle_y_history = deque(maxlen=5)

        self.star_detector = StarPatternDetector()

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        self.head_turn_history = deque(maxlen=30)
        self.shoulder_x_diff_history = deque(maxlen=30)

        # 히스토리 저장용
        self.hand_y_history = deque(maxlen=10)
        self.hand_x_history = deque(maxlen=10)
        self.hand_z_history = deque(maxlen=15)
        self.face_x_history = deque(maxlen=15)
        self.movement_magnitude = deque(maxlen=15)
        self.trail = deque(maxlen=60)
        self.hip_x_history = deque(maxlen=10)
        self.nose_y_history = deque(maxlen=10)

        self.cooldowns = {}  # 동작별 쿨다운 타이머

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def is_cooldown(self, motion):
        if self.cooldowns.get(motion, 0) > 0:
            self.cooldowns[motion] -= 1
            return True
        return False

    def detect_circle(self):
        if len(self.trail) < 30:
            return False
        xs, ys = zip(*self.trail)
        center_x, center_y = np.mean(xs), np.mean(ys)
        radii = [math.hypot(x - center_x, y - center_y) for x, y in self.trail]
        return np.var(radii) < 0.0005

    def count_direction_changes(self):
        if len(self.trail) < 10:
            return 0
        count = 0
        for i in range(2, len(self.trail)):
            x1, y1 = self.trail[i-2]
            x2, y2 = self.trail[i-1]
            x3, y3 = self.trail[i]
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            if dot < 0:
                count += 1
        return count
    
    def get_wrist_angle(self, hand):
        wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
        index_mcp = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        dx = index_mcp.x - wrist.x
        dy = index_mcp.y - wrist.y
        return math.degrees(math.atan2(dy, dx))

    def is_gun_pose(self, hand, is_left=True):
        lm = hand.landmark
        def extended(tip, pip): return lm[tip].y < lm[pip].y
        def bent(tip, pip): return lm[tip].y > lm[pip].y

        thumb_check = (
            lm[self.mp_hands.HandLandmark.THUMB_TIP].x < lm[self.mp_hands.HandLandmark.THUMB_IP].x
        ) if is_left else (
            lm[self.mp_hands.HandLandmark.THUMB_TIP].x > lm[self.mp_hands.HandLandmark.THUMB_IP].x
        )

        return (
            thumb_check and
            extended(self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP) and
            bent(self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
            bent(self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP) and
            bent(self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        )


    def detect(self, frame_rgb):
        motions = set()
        hand_result = self.hands.process(frame_rgb)
        face_result = self.face_mesh.process(frame_rgb)
        pose_result = self.pose.process(frame_rgb)
        mp_pose = self.mp_pose 

        if not hand_result.multi_hand_landmarks:
            return motions

        hands = hand_result.multi_hand_landmarks
        main_hand = hands[0]
        index_tip = main_hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = main_hand.landmark[self.mp_hands.HandLandmark.WRIST]

        self.hand_y_history.append(index_tip.y)
        self.hand_x_history.append(index_tip.x)
        self.hand_z_history.append(index_tip.z)
        self.trail.append((index_tip.x, index_tip.y))
        self.wrist_y_history.append(wrist.y)

        # if self.is_gun_pose(main_hand, is_left=True):
        #     angle = self.get_wrist_angle(main_hand)
        #     self.wrist_angle_history.append(angle)

        #     if len(self.wrist_angle_history) == 5 and len(self.wrist_y_history) == 5:
        #         delta_angle = self.wrist_angle_history[-1] - self.wrist_angle_history[0]
        #         delta_y = self.wrist_y_history[0] - self.wrist_y_history[-1]
        #         if 10 < delta_angle < 90 and delta_y > 0.03 and not self.is_cooldown("bang"):
        #             motions.add("bang, 총 빵야")
        #             self.cooldowns["bang"] = 30

        if len(self.hand_y_history) == 10:
            amp = max(self.hand_y_history) - min(self.hand_y_history)
            if amp > 0.2 and not self.is_cooldown("swing_updown"):
                motions.add("swing_updown, 팔 위아래 흔들기")
                self.cooldowns["swing_updown"] = 30

        # if len(self.hand_x_history) == 10:
        #     amp = max(self.hand_x_history) - min(self.hand_x_history)
        #     if amp > 0.2 and not self.is_cooldown("swing_leftright"):
        #         motions.add("swing_leftright, 손 좌우 흔들기")
        #         self.cooldowns["swing_leftright"] = 30

        # if len(self.hand_z_history) == 15:
        #     dz = self.hand_z_history[0] - self.hand_z_history[-1]
        #     if dz > 0.15 and not self.is_cooldown("hand_forward"):
        #         motions.add("hand_forward, 손 앞으로 뻗기")
        #         self.cooldowns["hand_forward"] = 30

        # if len(hands) == 2:
        #     wrist1 = hands[0].landmark[self.mp_hands.HandLandmark.WRIST]
        #     wrist2 = hands[1].landmark[self.mp_hands.HandLandmark.WRIST]
        #     dist = self.calculate_distance(wrist1, wrist2)
        #     self.movement_magnitude.append(dist)
        #     if len(self.movement_magnitude) == 15:
        #         delta = self.movement_magnitude[0] - self.movement_magnitude[-1]
        #         if delta > 0.15 and not self.is_cooldown("hands_gather"):
        #             motions.add("hands_gather, 양손 모으기")
        #             self.cooldowns["hands_gather"] = 30
        #         elif delta < -0.15 and not self.is_cooldown("hands_spread"):
        #             motions.add("hands_spread, 양손 벌리기")
        #             self.cooldowns["hands_spread"] = 30

            # tip1 = hands[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # tip2 = hands[1].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # if self.calculate_distance(tip1, tip2) < 0.05 and not self.is_cooldown("clap"):
            #     motions.add("clap, 박수")
            #     self.cooldowns["clap"] = 30

            # thumb1 = hands[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            # thumb2 = hands[1].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            # if self.calculate_distance(thumb1, thumb2) < 0.08 and not self.is_cooldown("heart"):
            #     motions.add("heart, 하트")
            #     self.cooldowns["heart"] = 30

        # if len(self.hand_y_history) == 10:
        #     dy = self.hand_y_history[-1] - self.hand_y_history[0]
        #     if dy > 0.15 and not self.is_cooldown("volume_down"):
        #         motions.add("volume_down, 손 아래로")
        #         self.cooldowns["volume_down"] = 30
        #     elif dy < -0.15 and not self.is_cooldown("volume_up"):
        #         motions.add("volume_up, 손 위로")
        #         self.cooldowns["volume_up"] = 30

        if len(hands) == 1:
            lm = hands[0].landmark
            index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
            if (self.calculate_distance(index_tip, middle_tip) > 0.08 and
                self.calculate_distance(index_tip, thumb_tip) > 0.08 and
                not self.is_cooldown("v_pose")):
                motions.add("v_pose, 브이 포즈")
                self.cooldowns["v_pose"] = 30

        # if face_result.multi_face_landmarks:
        #     nose = face_result.multi_face_landmarks[0].landmark[1]
        #     self.face_x_history.append(nose.x)
        #     if len(self.face_x_history) == 15:
        #         dx = self.face_x_history[-1] - self.face_x_history[0]
        #         if dx > 0.07 and not self.is_cooldown("turn_right"):
        #             motions.add("turn_right, 고개 오른쪽")
        #             self.cooldowns["turn_right"] = 30
        #         elif dx < -0.07 and not self.is_cooldown("turn_left"):
        #             motions.add("turn_left, 고개 왼쪽")
        #             self.cooldowns["turn_left"] = 30

        # if not self.is_cooldown("draw_circle") and self.detect_circle():
        #     motions.add("draw_circle, 원 그리기")
        #     self.cooldowns["draw_circle"] = 60

        # if not self.is_cooldown("draw_star") and self.count_direction_changes() >= 5:
        #     motions.add("draw_star, 별 그리기")
        #     self.cooldowns["draw_star"] = 60

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y

            if not hasattr(self, 'prev_wrist_order'):
                self.prev_wrist_order = None

            # current_order = 'L>R' if lw_y > rw_y else 'R>L'

            # if self.prev_wrist_order and self.prev_wrist_order != current_order:
            #     if not self.is_cooldown("drum_hit"):
            #         motions.add("drum_hit, 드럼치기")
            #         self.cooldowns["drum_hit"] = 30

            # self.prev_wrist_order = current_order

            # l_ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
            # self.left_ankle_y_history.append(l_ankle_y)

            # if len(self.left_ankle_y_history) == 5:
            #     delta_l = self.left_ankle_y_history[0] - self.left_ankle_y_history[-1]
            #     if delta_l > 0.08 and not self.is_cooldown("kick"):
            #         motions.add("kick, 킥 (왼발)")
            #         self.cooldowns["kick"] = 30

            # r_ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            # self.right_ankle_y_history.append(r_ankle_y)

            # if len(self.right_ankle_y_history) == 5:
            #     delta_r = self.right_ankle_y_history[0] - self.right_ankle_y_history[-1]
            #     if delta_r > 0.08 and not self.is_cooldown("kick"):
            #         motions.add("kick, 킥 (오른발)")
            #         self.cooldowns["kick"] = 30

            # l_knee_x = lm[self.mp_pose.PoseLandmark.LEFT_KNEE].x
            # r_knee_x = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
            # diff = abs(l_knee_x - r_knee_x)
            # self.knee_x_diff_history.append(diff)

            # if len(self.knee_x_diff_history) == 20:
            #     max_diff = max(self.knee_x_diff_history)
            #     min_diff = min(self.knee_x_diff_history)
            #     variation = max_diff - min_diff

            #     if variation > 0.12 and not self.is_cooldown("leg_dance"):
            #         motions.add("leg_dance, 개다리춤")
            #         self.cooldowns["leg_dance"] = 60

            # r_hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP].x
            # l_hip_x = lm[mp_pose.PoseLandmark.LEFT_HIP].x
            # self.hip_x_history.append((r_hip_x + l_hip_x) / 2)
            # if len(self.hip_x_history) == 10 and max(self.hip_x_history) - min(self.hip_x_history) > 0.1 and not self.is_cooldown("hip_shake"):
            #     motions.add("hip_shake, 엉덩이 흔들기")
            #     self.cooldowns["hip_shake"] = 30

            # dist = math.hypot(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x - lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
            #                   lm[mp_pose.PoseLandmark.RIGHT_WRIST].y - lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
            # if dist < 0.05 and not self.is_cooldown("hip_hit"):
            #     motions.add("hip_hit, 엉덩이 치기")
            #     self.cooldowns["hip_hit"] = 30

            # if lm[mp_pose.PoseLandmark.RIGHT_WRIST].y < lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y and not self.is_cooldown("lasso"):
            #     motions.add("lasso, 올가미")
            #     self.cooldowns["lasso"] = 30

            # if abs(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x - lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x) > 0.1 and not self.is_cooldown("arm_rotate"):
            #     motions.add("arm_rotate, 팔 돌리기")
            #     self.cooldowns["arm_rotate"] = 30

            self.nose_y_history.append(lm[mp_pose.PoseLandmark.NOSE].y)
            if len(self.nose_y_history) == 10 and max(self.nose_y_history) - min(self.nose_y_history) > 0.1 and not self.is_cooldown("squat"):
                motions.add("squat, 앉았다 일어났다")
                self.cooldowns["squat"] = 30
            
            lm = pose_result.pose_landmarks.landmark
            l_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
            r_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            diff = l_shoulder - r_shoulder
            self.shoulder_x_diff_history.append(diff)

            if len(self.shoulder_x_diff_history) == 30:
                min_diff = min(self.shoulder_x_diff_history)
                max_diff = max(self.shoulder_x_diff_history)

                if min_diff < -0.1 and max_diff > 0.1 and not self.is_cooldown("spin"):
                    motions.add("spin, 회전 동작 감지")
                    self.cooldowns["spin"] = 60

        # index_tip = main_hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        # self.star_detector.add_point(index_tip.x, index_tip.y)
        # if not self.is_cooldown("draw_star") and self.star_detector.detect_star_like_pattern():
        #     motions.add("draw_star, 별 그리기")
        #     self.cooldowns["draw_star"] = 60


        priority = [
            "spin", "clap", "heart", "hands_gather", "hands_spread",
            "hand_forward", "v_pose", "volume_up", "volume_down",
            "swing_updown", "swing_leftright", "turn_left", "turn_right",
            "draw_circle", "draw_star", "drum_hit", "kick", "leg_swing",
            "hip_shake", "hip_hit", "bang", "lasso", "arm_rotate", "squat"
        ]

        for motion in priority:
            for m in motions:
                if m.startswith(motion):
                    return {m}

        return motions