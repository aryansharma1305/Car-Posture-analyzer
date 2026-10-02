"""Golden-angle tests for geometry_utils.

Pure math, no camera needed. `signed_angle_with_vertical` is the primitive;
`angle_with_vertical` is its direction-invariant magnitude view, and the
equivalence between them is asserted here so they cannot drift apart.

Coordinate convention: OpenCV pixel space. x grows rightward, y grows DOWNWARD.
So a point with a smaller y is physically higher in the frame.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_utils import (  # noqa: E402
    angle_3pts,
    angle_with_vertical,
    fold_to_vertical_magnitude,
    is_measured,
    signed_angle_with_vertical,
)


def p(x, y):
    return np.array([float(x), float(y)])


# --------------------------------------------------------------------------
# angle_3pts: angle at B formed by A-B-C, 0..180
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, c, expected", [
    # Right angle: A directly above B, C directly right of B.
    (p(0, -100), p(0, 0), p(100, 0), 90.0),
    # Straight line through B.
    (p(0, -100), p(0, 0), p(0, 100), 180.0),
    # Fully folded back on itself.
    (p(0, -100), p(0, 0), p(0, -100), 0.0),
    # 45 degrees.
    (p(0, -100), p(0, 0), p(100, -100), 45.0),
    # A seated knee: thigh horizontal, shank vertical -> 90.
    (p(-100, 0), p(0, 0), p(0, 100), 90.0),
])
def test_angle_3pts_known_values(a, b, c, expected):
    assert angle_3pts(a, b, c) == pytest.approx(expected, abs=0.01)


def test_angle_3pts_is_symmetric_in_outer_points():
    a, b, c = p(10, -50), p(0, 0), p(80, 20)
    assert angle_3pts(a, b, c) == pytest.approx(angle_3pts(c, b, a), abs=1e-9)


def test_angle_3pts_is_scale_invariant():
    a, b, c = p(0, -100), p(0, 0), p(100, -100)
    big = [v * 7.5 for v in (a, b, c)]
    assert angle_3pts(a, b, c) == pytest.approx(angle_3pts(*big), abs=1e-6)


def test_angle_3pts_degenerate_points_are_nan_not_a_number():
    """Coincident landmarks happen when MediaPipe loses a joint.

    This used to return 0.0, which is a MEASUREMENT - a fully folded joint -
    and fired a knee-strain risk. NaN cannot be mistaken for one.
    """
    val = angle_3pts(p(0, 0), p(0, 0), p(0, 0))
    assert math.isnan(val)
    assert not is_measured(val)


# --------------------------------------------------------------------------
# angle_with_vertical: deviation of segment p1->p2 from the vertical axis
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p1, p2, expected", [
    # Perfectly vertical segment (hips below, shoulders above).
    (p(0, 100), p(0, 0), 0.0),
    # Perfectly horizontal -> 90 off vertical.
    (p(0, 0), p(100, 0), 90.0),
    # 45 degrees.
    (p(0, 100), p(100, 0), 45.0),
])
def test_angle_with_vertical_magnitude(p1, p2, expected):
    assert angle_with_vertical(p1, p2) == pytest.approx(expected, abs=0.01)


def test_magnitude_view_is_sign_blind_by_design():
    """angle_with_vertical folds direction away - that is its contract.

    Threshold rules and RULA/REBA want magnitude. Anything that cares about
    lean DIRECTION must read the signed function instead.
    """
    hip = p(100, 400)
    forward = p(130, 300)
    reclined = p(70, 300)
    assert angle_with_vertical(hip, forward) == pytest.approx(
        angle_with_vertical(hip, reclined), abs=1e-9
    )


# --------------------------------------------------------------------------
# signed_angle_with_vertical: 0 = straight up, + = toward image right
# --------------------------------------------------------------------------

def test_forward_lean_is_positive():
    assert signed_angle_with_vertical(p(100, 400), p(130, 300)) > 0


def test_recline_is_negative():
    assert signed_angle_with_vertical(p(100, 400), p(70, 300)) < 0


def test_forward_and_recline_are_mirror_images():
    hip = p(100, 400)
    assert signed_angle_with_vertical(hip, p(130, 300)) == pytest.approx(
        -signed_angle_with_vertical(hip, p(70, 300)), abs=1e-9
    )


@pytest.mark.parametrize("p1, p2, expected", [
    (p(0, 100), p(0, 0), 0.0),        # straight up
    (p(0, 100), p(100, 0), 45.0),     # up and to the right
    (p(0, 100), p(-100, 0), -45.0),   # up and to the left
    (p(0, 0), p(100, 0), 90.0),       # horizontal right
    (p(0, 0), p(-100, 0), -90.0),     # horizontal left
    (p(0, 0), p(0, 100), 180.0),      # straight down
])
def test_signed_angle_known_values(p1, p2, expected):
    assert signed_angle_with_vertical(p1, p2) == pytest.approx(expected, abs=0.01)


def test_signed_angle_degenerate_segment_is_nan_not_upright():
    """0.0 here would have meant "perfectly upright" and scored 100."""
    val = signed_angle_with_vertical(p(5, 5), p(5, 5))
    assert math.isnan(val)
    assert math.isnan(angle_with_vertical(p(5, 5), p(5, 5)))
    assert math.isnan(fold_to_vertical_magnitude(val))


# --------------------------------------------------------------------------
# The two views must never disagree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("signed, expected", [
    (0.0, 0.0), (16.7, 16.7), (-16.7, 16.7),
    (90.0, 90.0), (-90.0, 90.0), (180.0, 0.0), (-180.0, 0.0),
])
def test_fold_to_vertical_magnitude(signed, expected):
    assert fold_to_vertical_magnitude(signed) == pytest.approx(expected, abs=1e-9)


def test_unsigned_is_exactly_the_folded_signed_value():
    """One computation of record - asserted over a dense sweep of directions."""
    rng = np.random.default_rng(1234)
    for _ in range(2000):
        p1 = rng.uniform(-500, 500, 2)
        p2 = rng.uniform(-500, 500, 2)
        if np.linalg.norm(p2 - p1) < 1e-3:
            continue
        folded = fold_to_vertical_magnitude(signed_angle_with_vertical(p1, p2))
        assert folded == pytest.approx(angle_with_vertical(p1, p2), abs=1e-12)


def test_unsigned_stays_within_its_documented_range():
    rng = np.random.default_rng(99)
    for _ in range(2000):
        p1 = rng.uniform(-500, 500, 2)
        p2 = rng.uniform(-500, 500, 2)
        assert 0.0 <= angle_with_vertical(p1, p2) <= 90.0
