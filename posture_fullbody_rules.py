"""Full-body state detection and upper-body labelling.

The upper-body label thresholds are NOT defined here - they come from a
driver_model.PostureReference, because the correct threshold depends on whether
the subject is at a desk or in a car seat. This module used to hardcode the
desk numbers (18/20/35), which labelled every correctly reclined driver as
"Slouch".
"""
from dataclasses import dataclass
import numpy as np

from driver_model import DEFAULT_REFERENCE, PostureReference, torso_view
from geometry_utils import is_measured
from landmark_confidence import VISIBILITY_THRESHOLD

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

    # Visibility threshold (MediaPipe landmark.visibility). Sourced from
    # landmark_confidence so this module and posture_angles cannot disagree
    # about which landmarks are trustworthy - they used to carry 0.5 separately,
    # and only this one was ever consulted.
    vis_thr: float = VISIBILITY_THRESHOLD

def classify_upper_body(angles: dict,
                       reference: PostureReference = DEFAULT_REFERENCE) -> str:
    """Upper-body only: Neutral / Slouch / ForwardHead / RaisedShoulders.

    Which angle view each threshold reads is the reference's call. Under
    DRIVING, slouch is judged on the SIGNED trunk angle, so recline against the
    backrest is not slouch; under DESK it is judged on the magnitude, where any
    departure from vertical counts.

    A threshold whose angle view is not available in `angles` - asking for a
    signed value in a replayed legacy log - is skipped rather than guessed.
    """
    trunk = torso_view(angles, reference.slouch_key)
    if trunk is not None and trunk > reference.slouch_deg:
        return "Slouch"

    neck = torso_view(angles, reference.forward_head_key)
    if neck is not None and neck > reference.forward_head_deg:
        return "ForwardHead"

    shoulders = [angles[k] for k in ("left_shoulder_elev", "right_shoulder_elev")
                 if is_measured(angles.get(k))]
    if shoulders and max(shoulders) > reference.shoulder_elev_deg:
        return "RaisedShoulders"

    # Nothing the reference judges was measurable, so "Neutral" would be a
    # claim rather than an observation.
    if trunk is None and neck is None and not shoulders:
        return "Unknown"

    return "Neutral"

def detect_body_state(pts: dict, angles: dict, vis: dict, img_size, cfg: BodyConfig = BodyConfig()):
    """
    Returns ("Standing"|"Sitting"|"Lying"|"Squatting"|"Unknown", extras_dict)
    Degrades gracefully when legs are not visible.
    """
    h, w = img_size
    trunk = torso_view(angles, "trunk_from_vertical")

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

    # Angles. An unmeasured joint is dropped rather than compared: NaN fails
    # every comparison, so a lost knee would have read as "not bent" and a lost
    # trunk as "not upright and not horizontal" - both silently plausible.
    knees = [angles[k] for k in ("left_knee_angle", "right_knee_angle")
             if is_measured(angles.get(k))]
    hips = [angles[k] for k in ("left_hip_angle", "right_hip_angle")
            if is_measured(angles.get(k))]
    knee_min = min(knees) if knees else None
    hip_min = min(hips) if hips else None

    # Legs count as visible only if MediaPipe saw them AND the angles built on
    # them survived masking.
    lower_vis = lower_vis and knee_min is not None and hip_min is not None

    if trunk is None:
        # Without a torso line there is no body state to infer. Everything
        # below branches on it.
        return "Unknown", {
            "upright_deg": None,
            "bbox_aspect": aspect,
            "lower_visible": lower_vis,
            "lean_side": lean_side,
            "debug": "trunk angle not measured",
        }

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
            # No legs visible -> cannot decide between sit/stand; fall back to
            # torso cues. (`horizontal` is False in this branch by construction,
            # so the lying case is handled above, not here.)
            if upright:
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
