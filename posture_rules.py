from dataclasses import dataclass

@dataclass
class PostureConfig:
    # Seated detection
    hip_angle_target: float = 95.0     # ideal ~90
    hip_angle_tol: float = 25.0        # tolerance
    knee_angle_target: float = 95.0
    knee_angle_tol: float = 25.0

    # Labels
    trunk_slouch_deg: float = 18.0     # trunk tilt from vertical
    neck_forward_deg: float = 20.0     # neck tilt from vertical
    shoulder_elev_deg: float = 35.0    # raised arms typing/phone

    # Asymmetry (lean)
    trunk_lean_diff_deg: float = 12.0  # left vs right hip/shoulder height heuristic (not used here)
    knee_mismatch_deg: float = 20.0    # big mismatch suggests leg cross

def is_seated(angles: dict, cfg: PostureConfig) -> bool:
    hips_ok = (
        abs(angles["left_hip_angle"]  - cfg.hip_angle_target)  <= cfg.hip_angle_tol or
        abs(angles["right_hip_angle"] - cfg.hip_angle_target)  <= cfg.hip_angle_tol
    )
    knees_ok = (
        abs(angles["left_knee_angle"] - cfg.knee_angle_target) <= cfg.knee_angle_tol and
        abs(angles["right_knee_angle"]- cfg.knee_angle_target) <= cfg.knee_angle_tol
    )
    return hips_ok and knees_ok

def classify_posture(angles: dict, cfg: PostureConfig = PostureConfig()):
    """
    Returns (primary_label, tags) where:
      - primary_label in {"Neutral", "Slouch", "ForwardHead", "RaisedShoulders", "CrossedLegs", "NotSeated"}
      - tags is a set of secondary flags for UI (e.g. {"mild"})
    """
    tags = set()

    if not is_seated(angles, cfg):
        return "NotSeated", tags

    trunk = angles["trunk_from_vertical"]
    neck  = angles["neck_from_vertical"]

    # Crossed legs heuristic: knees differ a lot from ~90, or left/right mismatch big
    knee_avg = 0.5 * (angles["left_knee_angle"] + angles["right_knee_angle"])
    knee_diff = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    crossed = (abs(knee_avg - cfg.knee_angle_target) > cfg.knee_angle_tol and knee_diff > cfg.knee_mismatch_deg)

    raised_shoulders = (
        angles["left_shoulder_elev"]  > cfg.shoulder_elev_deg or
        angles["right_shoulder_elev"] > cfg.shoulder_elev_deg
    )

    # Priority order: Slouch > ForwardHead > CrossedLegs > RaisedShoulders > Neutral
    if trunk > cfg.trunk_slouch_deg:
        if trunk < cfg.trunk_slouch_deg + 8: tags.add("mild")
        return "Slouch", tags

    if neck > cfg.neck_forward_deg:
        if neck < cfg.neck_forward_deg + 8: tags.add("mild")
        return "ForwardHead", tags

    if crossed:
        return "CrossedLegs", tags

    if raised_shoulders:
        return "RaisedShoulders", tags

    return "Neutral", tags
