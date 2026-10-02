"""Which landmarks an angle depends on, and whether they can be trusted.

MediaPipe emits a landmark for every joint on every frame, whether or not it
can see one. A driver whose legs are under the dash still gets knee and hip
coordinates - extrapolated from the torso - and before this module those
extrapolations flowed into the comfort score as measurements. The only signal
that they were guesses was `landmark.visibility`, and the pipeline dropped it:
pose_core kept the coordinates and discarded the confidence, and the two
monitors each rebuilt a partial visibility dict by hardcoding eight landmark
indices.

So this module owns three things that were previously implicit or duplicated:

  VISIBILITY_THRESHOLD   the one definition of "trustworthy"
  ANGLE_LANDMARKS        which landmarks each angle is computed from - knowledge
                         that lived only inside posture_angles.compute_angles
  angle_confidence()     per-angle confidence, as the minimum over the
                         landmarks it needs

Minimum, not mean: an angle is only as good as its worst input. Averaging would
let two confident landmarks carry a third that MediaPipe invented.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

# A landmark at or below this visibility is not trusted. 0.5 is MediaPipe's own
# conventional midpoint and matches the value posture_fullbody_rules.BodyConfig
# had been carrying as vis_thr - which was the only place in the project that
# consulted visibility at all.
VISIBILITY_THRESHOLD = 0.5

# The landmarks posture_angles.compute_angles reads for each angle it emits.
# Kept here rather than in that function so confidence can be computed BEFORE
# the angles, and so a reader can see what an angle actually depends on.
ANGLE_LANDMARKS: Dict[str, frozenset] = {
    # Torso: hip centre -> shoulder centre.
    "trunk_signed": frozenset({"left_hip", "right_hip",
                               "left_shoulder", "right_shoulder"}),
    # Neck: shoulder centre -> head reference (nose averaged with both ears).
    "neck_signed": frozenset({"left_shoulder", "right_shoulder",
                              "nose", "left_ear", "right_ear"}),
    "left_hip_angle": frozenset({"left_shoulder", "left_hip", "left_knee"}),
    "right_hip_angle": frozenset({"right_shoulder", "right_hip", "right_knee"}),
    "left_knee_angle": frozenset({"left_hip", "left_knee", "left_ankle"}),
    "right_knee_angle": frozenset({"right_hip", "right_knee", "right_ankle"}),
    "left_shoulder_elev": frozenset({"left_shoulder", "left_elbow"}),
    "right_shoulder_elev": frozenset({"right_shoulder", "right_elbow"}),
}

# The magnitude views are derived from the signed ones, so they inherit their
# confidence exactly. Listed rather than computed so the mapping is one fact in
# one place; posture_angles.MAGNITUDE_FROM_SIGNED is its mirror.
DERIVED_ANGLES = {
    "trunk_from_vertical": "trunk_signed",
    "neck_from_vertical": "neck_signed",
}


def angle_confidence(
    visibility: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Confidence 0..1 for every angle, as the minimum over its landmarks.

    A landmark missing from `visibility` is treated as unknown and scores 0.0:
    an angle whose inputs were never reported cannot be more trustworthy than
    one whose inputs were reported as invisible. `None` means the caller has no
    visibility information at all, which yields an empty dict - callers then
    apply no masking rather than masking everything.
    """
    if visibility is None:
        return {}
    confidence: Dict[str, float] = {}
    for angle, landmarks in ANGLE_LANDMARKS.items():
        confidence[angle] = min(
            float(visibility.get(landmark, 0.0)) for landmark in landmarks
        )
    for derived, source in DERIVED_ANGLES.items():
        confidence[derived] = confidence[source]
    return confidence


def unmeasurable_angles(
    visibility: Optional[Dict[str, float]],
    threshold: float = VISIBILITY_THRESHOLD,
) -> Set[str]:
    """Angle names whose least-visible landmark is at or below `threshold`.

    Empty when `visibility` is None, so a caller with no confidence
    information gets the old unmasked behaviour rather than a blank frame.
    """
    return {
        angle
        for angle, score in angle_confidence(visibility).items()
        if score <= threshold
    }


def frame_confidence(
    visibility: Optional[Dict[str, float]],
    angles: Optional[Iterable[str]] = None,
) -> Optional[float]:
    """One summary number for a frame: the worst confidence among `angles`.

    Defaults to every angle in ANGLE_LANDMARKS. Returns None when there is no
    visibility information, which is written to the log as empty rather than as
    a confident zero.
    """
    scores = angle_confidence(visibility)
    if not scores:
        return None
    wanted = [a for a in (angles or ANGLE_LANDMARKS) if a in scores]
    if not wanted:
        return None
    return min(scores[a] for a in wanted)
