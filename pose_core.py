import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseExtractor:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_bgr(self, frame_bgr):
        """Returns (results, pts_dict, (h,w)) or (None, None, None) if no pose."""
        h, w = frame_bgr.shape[:2]
        res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None, None, None
        lm = res.pose_landmarks.landmark

        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=float)

        pts = {
            "nose": px(mp_pose.PoseLandmark.NOSE),
            "left_ear": px(mp_pose.PoseLandmark.LEFT_EAR),
            "right_ear": px(mp_pose.PoseLandmark.RIGHT_EAR),
            "left_shoulder": px(mp_pose.PoseLandmark.LEFT_SHOULDER),
            "right_shoulder": px(mp_pose.PoseLandmark.RIGHT_SHOULDER),
            "left_elbow": px(mp_pose.PoseLandmark.LEFT_ELBOW),
            "right_elbow": px(mp_pose.PoseLandmark.RIGHT_ELBOW),
            "left_wrist": px(mp_pose.PoseLandmark.LEFT_WRIST),
            "right_wrist": px(mp_pose.PoseLandmark.RIGHT_WRIST),
            "left_hip": px(mp_pose.PoseLandmark.LEFT_HIP),
            "right_hip": px(mp_pose.PoseLandmark.RIGHT_HIP),
            "left_knee": px(mp_pose.PoseLandmark.LEFT_KNEE),
            "right_knee": px(mp_pose.PoseLandmark.RIGHT_KNEE),
            "left_ankle": px(mp_pose.PoseLandmark.LEFT_ANKLE),
            "right_ankle": px(mp_pose.PoseLandmark.RIGHT_ANKLE),
        }
        return res, pts, (h, w)

    @staticmethod
    def draw(frame_bgr, results):
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
        )
