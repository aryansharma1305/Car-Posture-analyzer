"""Seated-posture labelling.

Thresholds come from a driver_model.PostureReference. PostureConfig stays as the
call-site knob it always was, but its defaults are now DERIVED from the
reference instead of being a second copy of the same numbers that could drift
away from posture_fullbody_rules and ergonomics_scores - which is exactly what
had happened.
"""
from dataclasses import dataclass

from driver_model import (
    DEFAULT_REFERENCE,
    MILD_BAND_DEG,
    PostureReference,
    Range,
    torso_view,
)


@dataclass
class PostureConfig:
    """Thresholds for one seating context.

    Build it from a reference with `PostureConfig.from_reference(DRIVING)`; the
    no-argument default is the DEFAULT_REFERENCE, which preserves this module's
    historical desk behaviour.
    """
    # Seated detection, as target +- tolerance
    hip_angle_target: float = 95.0
    hip_angle_tol: float = 25.0
    knee_angle_target: float = 95.0
    knee_angle_tol: float = 25.0

    # Labels
    slouch_key: str = DEFAULT_REFERENCE.slouch_key
    trunk_slouch_deg: float = DEFAULT_REFERENCE.slouch_deg
    forward_head_key: str = DEFAULT_REFERENCE.forward_head_key
    neck_forward_deg: float = DEFAULT_REFERENCE.forward_head_deg
    shoulder_elev_deg: float = DEFAULT_REFERENCE.shoulder_elev_deg

    # Width of the band above a threshold still tagged 'mild'.
    mild_band_deg: float = MILD_BAND_DEG

    # Asymmetry (lean)
    trunk_lean_diff_deg: float = 12.0  # left vs right hip/shoulder height heuristic (not used here)
    knee_mismatch_deg: float = 20.0    # big mismatch suggests leg cross

    @classmethod
    def from_reference(cls, reference: PostureReference) -> "PostureConfig":
        """Thresholds for `reference`, with seated detection from its
        seated_hip/seated_knee ranges expressed as target +- tolerance."""
        def target_tol(rng: Range):
            return (rng.min + rng.max) / 2.0, (rng.max - rng.min) / 2.0

        hip_target, hip_tol = target_tol(reference.seated_hip)
        knee_target, knee_tol = target_tol(reference.seated_knee)
        return cls(
            hip_angle_target=hip_target,
            hip_angle_tol=hip_tol,
            knee_angle_target=knee_target,
            knee_angle_tol=knee_tol,
            slouch_key=reference.slouch_key,
            trunk_slouch_deg=reference.slouch_deg,
            forward_head_key=reference.forward_head_key,
            neck_forward_deg=reference.forward_head_deg,
            shoulder_elev_deg=reference.shoulder_elev_deg,
        )


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

    The trunk and neck thresholds read whichever angle view cfg names - signed
    for a driving config, magnitude for a desk one. A view the caller did not
    supply is skipped, not guessed.
    """
    tags = set()

    if not is_seated(angles, cfg):
        return "NotSeated", tags

    trunk = torso_view(angles, cfg.slouch_key)
    neck  = torso_view(angles, cfg.forward_head_key)

    # Crossed legs heuristic: knees differ a lot from ~90, or left/right mismatch big
    knee_avg = 0.5 * (angles["left_knee_angle"] + angles["right_knee_angle"])
    knee_diff = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    crossed = (abs(knee_avg - cfg.knee_angle_target) > cfg.knee_angle_tol and knee_diff > cfg.knee_mismatch_deg)

    raised_shoulders = (
        angles.get("left_shoulder_elev", 0.0)  > cfg.shoulder_elev_deg or
        angles.get("right_shoulder_elev", 0.0) > cfg.shoulder_elev_deg
    )

    # Priority order: Slouch > ForwardHead > CrossedLegs > RaisedShoulders > Neutral
    if trunk is not None and trunk > cfg.trunk_slouch_deg:
        if trunk < cfg.trunk_slouch_deg + cfg.mild_band_deg: tags.add("mild")
        return "Slouch", tags

    if neck is not None and neck > cfg.neck_forward_deg:
        if neck < cfg.neck_forward_deg + cfg.mild_band_deg: tags.add("mild")
        return "ForwardHead", tags

    if crossed:
        return "CrossedLegs", tags

    if raised_shoulders:
        return "RaisedShoulders", tags

    return "Neutral", tags
