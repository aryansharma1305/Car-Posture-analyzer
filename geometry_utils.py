import numpy as np

def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> float:
    """
    Angle at point B formed by A-B-C in degrees (0..180).
    """
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba) + eps
    nc = np.linalg.norm(bc) + eps
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

import numpy as np

def angle_with_vertical(p1, p2, eps: float = 1e-6) -> float:
    """
    Angle (0..90) between segment p1->p2 and the vertical axis.
    Up vs down are treated the same (direction-invariant).
    """
    v = p2 - p1
    nv = np.linalg.norm(v) + eps
    vert = np.array([0.0, 1.0], dtype=float)
    cosang = np.dot(v / nv, vert)
    cosang = np.clip(abs(cosang), -1.0, 1.0)  # abs = ignore direction
    return float(np.degrees(np.arccos(cosang)))
