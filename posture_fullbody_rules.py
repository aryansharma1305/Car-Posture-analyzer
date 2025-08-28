from dataclasses import dataclass
import numpy as np

@dataclass
class BodyConfig:
    # Upright/horizontal (from trunk vertical angle)
    upright_deg: float = 25.0      # <= this -> upright
    horizontal_deg: float = 65.0   # >= this -> horizontal (lying)

    # Joints (simple ranges)
    knee_straight_deg: float = 160.0
    hip_straight_deg: float  = 160.0
    sit_min_deg: float = 70.0      # 70..120 ~ "bent"
    sit_max_deg: float = 120.0

    # Lateral lean (shoulder center vs hip center horizontal offset)
    side_lean_ratio: float = 0.35  # fraction of shoulder width

    # Visibility threshold (MediaPipe landmark.visibility)
    vis_thr: float = 0.5

def classify_upper_body(angles: dict) -> str:
    """Upper-body only: Neutral / Slouch / ForwardHead / RaisedShoulders."""
    if angles["trunk_from_vertical"] > 18.0:
        return "Slouch"
    if angles["neck_from_vertical"] > 20.0:
        return "ForwardHead"
    if angles["left_shoulder_elev"] > 35.0 or angles["right_shoulder_elev"] > 35.0:
        return "RaisedShoulders"
    return "Neutral"

def detect_body_state(pts: dict, angles: dict, vis: dict, img_size, cfg: BodyConfig = BodyConfig()):
    """
    Returns ("Standing"|"Sitting"|"Lying"|"Squatting"|"Unknown", extras_dict)
    Degrades gracefully when legs are not visible.
    """
    h, w = img_size
    trunk = angles["trunk_from_vertical"]

    # Shoulder/hip centers & shoulder width
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    lh, rh = pts["left_hip"], pts["right_hip"]
    shoulder_c = (ls + rs) / 2.0
    hip_c = (lh + rh) / 2.0
    shoulder_w = max(1.0, abs(ls[0] - rs[0]))

    # Lateral lean (left/right)
    dx = shoulder_c[0] - hip_c[0]
    lean_side = None
    if abs(dx) > cfg.side_lean_ratio * shoulder_w:
        lean_side = "LeanLeft" if dx < 0 else "LeanRight"  # from camera view

    # Compute bbox aspect (use shoulders/hips/knees/ankles if available)
    keys = ["left_shoulder","right_shoulder","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]
    xy = np.vstack([pts[k] for k in keys if k in pts])
    bbox_w = float(np.max(xy[:,0]) - np.min(xy[:,0])) if xy.size else 0.0
    bbox_h = float(np.max(xy[:,1]) - np.min(xy[:,1])) if xy.size else 0.0
    aspect = (bbox_w / max(1.0, bbox_h))

    # Visibility flags
    knees_vis = (vis.get("left_knee",0) > cfg.vis_thr and vis.get("right_knee",0) > cfg.vis_thr)
    ankles_vis = (vis.get("left_ankle",0) > cfg.vis_thr and vis.get("right_ankle",0) > cfg.vis_thr)
    hips_vis = (vis.get("left_hip",0) > cfg.vis_thr and vis.get("right_hip",0) > cfg.vis_thr)
    lower_vis = (knees_vis or ankles_vis) and hips_vis

    # Angles
    knee_min = min(angles["left_knee_angle"], angles["right_knee_angle"])
    hip_min  = min(angles["left_hip_angle"],  angles["right_hip_angle"])

    upright = (trunk <= cfg.upright_deg)
    horizontal = (trunk >= cfg.horizontal_deg)

    # ---------------------------
    # Full-body primary state
    # ---------------------------
    state = "Unknown"
    reason = ""

    if horizontal:
        # Wide & flat + horizontal spine -> likely lying
        if aspect > 1.3:
            state = "Lying"
            reason = f"horizontal:{trunk:.1f} aspect:{aspect:.2f}"
        else:
            state = "Unknown"; reason = f"horizontal:{trunk:.1f} but aspect:{aspect:.2f}"
    else:
        if lower_vis:
            # Standing: straight hips/knees + upright
            if upright and knee_min >= cfg.knee_straight_deg and hip_min >= cfg.hip_straight_deg:
                state = "Standing"; reason = f"upright:{trunk:.1f} knees:{knee_min:.1f} hips:{hip_min:.1f}"
            # Sitting: hips & knees ~ 70..120 + upright-ish
            elif (cfg.sit_min_deg <= knee_min <= cfg.sit_max_deg) and \
                 (cfg.sit_min_deg <= hip_min  <= cfg.sit_max_deg) and \
                 (trunk <= 40.0):
                state = "Sitting"; reason = f"knee:{knee_min:.1f} hip:{hip_min:.1f} trunk:{trunk:.1f}"
            # Squatting/Kneeling: very bent knees/hips but still upright-ish
            elif knee_min < cfg.sit_min_deg and trunk <= 45.0:
                state = "Squatting"; reason = f"knee:{knee_min:.1f} trunk:{trunk:.1f}"
            else:
                state = "Unknown"; reason = f"angles inconclusive k:{knee_min:.1f} h:{hip_min:.1f} trunk:{trunk:.1f}"
        else:
            # No legs visible -> cannot decide between sit/stand; fall back to torso cues
            if horizontal:
                state = "Lying"; reason = "horizontal trunk without legs"
            elif upright:
                state = "Unknown"; reason = "upper-body only (upright)"
            else:
                state = "Unknown"; reason = "upper-body only (tilted)"

    extras = {
        "upright_deg": trunk,
        "bbox_aspect": aspect,
        "lower_visible": lower_vis,
        "lean_side": lean_side,
        "debug": reason
    }
    return state, extras
