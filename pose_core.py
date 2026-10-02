"""MediaPipe pose extraction.

The landmark set below is the project's vocabulary: every name here is what
landmark_confidence.ANGLE_LANDMARKS and posture_angles.compute_angles refer to.
LANDMARK_INDEX is the single index map - both monitors used to carry their own
partial copy covering eight landmarks, so the confidence of the head, elbow and
wrist landmarks was never available to anything.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# name -> MediaPipe PoseLandmark. Defines both the pts dict and the visibility
# dict, so the two can never cover different landmarks.
LANDMARK_INDEX = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
    "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

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

        pts = {name: px(index) for name, index in LANDMARK_INDEX.items()}
        return res, pts, (h, w)

    @staticmethod
    def visibility(results) -> dict:
        """MediaPipe's own confidence per landmark, 0..1, keyed by our names.

        This is the only signal distinguishing a landmark MediaPipe SAW from one
        it extrapolated behind an occlusion, and both are returned in `pts` as
        ordinary coordinates. Feed it to posture_angles.compute_angles so the
        angles built on invented landmarks come back as NaN.
        """
        if results is None or not results.pose_landmarks:
            return {}
        lm = results.pose_landmarks.landmark
        return {name: float(lm[index].visibility)
                for name, index in LANDMARK_INDEX.items()}

    @staticmethod
    def draw(frame_bgr, results):
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
        )
