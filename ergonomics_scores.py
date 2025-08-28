from typing import Dict, Tuple, List

# Colors as BGR tuples (for convenience if UI wants them)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


def _bucket(value: float, bounds: Tuple[float, float]) -> int:
    """Return 1/2/3 for value within simple thresholds (lo, hi)."""
    lo, hi = bounds
    if value <= lo:
        return 1
    if value <= hi:
        return 2
    return 3


def compute_rula(angles: Dict[str, float]) -> Tuple[int, str]:
    """Very simplified RULA category (1..3) and label.
    Uses neck, trunk and shoulder elevation.
    """
    neck = _bucket(angles.get("neck_from_vertical", 0.0), (10.0, 20.0))
    trunk = _bucket(angles.get("trunk_from_vertical", 0.0), (10.0, 20.0))
    sh = max(
        _bucket(angles.get("left_shoulder_elev", 0.0), (20.0, 45.0)),
        _bucket(angles.get("right_shoulder_elev", 0.0), (20.0, 45.0)),
    )
    score = neck + trunk + sh
    if score <= 3:
        return 1, "Acceptable"
    if score <= 5:
        return 2, "Investigate"
    return 3, "High Risk"


def compute_reba(angles: Dict[str, float]) -> Tuple[int, str]:
    """Very simplified REBA category (1..3) and label.
    Uses trunk tilt and leg posture (knee/hip flexion).
    """
    trunk_cat = _bucket(angles.get("trunk_from_vertical", 0.0), (10.0, 20.0))
    knee_min = min(angles.get("left_knee_angle", 180.0), angles.get("right_knee_angle", 180.0))
    hip_min = min(angles.get("left_hip_angle", 180.0), angles.get("right_hip_angle", 180.0))

    # Knee/hip flexion buckets: near 180 -> 1, 70..120 -> 2, <70 -> 3
    def flex_cat(a: float) -> int:
        if a >= 150.0:
            return 1
        if a >= 70.0:
            return 2
        return 3

    leg_cat = max(flex_cat(knee_min), flex_cat(hip_min))
    score = trunk_cat + leg_cat
    if score <= 2:
        return 1, "Low Risk"
    if score <= 4:
        return 2, "Medium Risk"
    return 3, "High Risk"


def feedback_lines(angles: Dict[str, float]) -> List[str]:
    """Generate short feedback lines similar to screenshots."""
    lines: List[str] = []

    trunk = angles.get("trunk_from_vertical", 0.0)
    neck = angles.get("neck_from_vertical", 0.0)
    hip_avg = 0.5 * (angles.get("left_hip_angle", 0.0) + angles.get("right_hip_angle", 0.0))
    knee_avg = 0.5 * (angles.get("left_knee_angle", 0.0) + angles.get("right_knee_angle", 0.0))
    sh_max = max(angles.get("left_shoulder_elev", 0.0), angles.get("right_shoulder_elev", 0.0))

    if trunk > 26.0:
        lines.append("Trunk: Severe slouching detected.")
    elif trunk > 18.0:
        lines.append("Trunk: Mild slouching detected.")

    if neck > 25.0:
        lines.append("Neck: Forward head posture.")

    if sh_max > 35.0:
        lines.append("Shoulders: Elevated. Relax and lower them.")

    if hip_avg > 110.0 or hip_avg < 80.0:
        lines.append("Hips: Poor lower back support (check seat).")

    if knee_avg < 80.0:
        lines.append("Knees: Too closed; lower chair or move feet forward.")

    if not lines:
        lines.append("All Clear: Good posture.")

    return lines


def risk_buckets(rula_cat: int, reba_cat: int) -> str:
    """Return 'good' | 'neutral' | 'high' from categories."""
    if rula_cat == 1 and reba_cat == 1:
        return "good"
    if rula_cat == 3 or reba_cat == 3:
        return "high"
    return "neutral"
