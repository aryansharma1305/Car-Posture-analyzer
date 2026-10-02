"""Exponential smoothing of the angle series.

NaN handling is the whole subtlety here. An unmeasured angle arrives as NaN
(geometry_utils.DEGENERATE_ANGLE), and `alpha * nan + (1 - alpha) * state` is
NaN - so a single lost landmark would poison a filter permanently, and every
later frame would report NaN even once the joint came back into view.

So an unmeasured sample does two things and only two things: it leaves the
filter state untouched, and it is passed through as NaN. Passing through rather
than substituting the last good value is deliberate - the caller asked what the
angle is NOW, and the honest answer is "unknown". A UI that wants continuity can
read last_good() and show it as stale; the log must not, which is why
session_schema writes an empty cell for it.
"""
from geometry_utils import is_measured


class EMA:
    """Exponential moving average for scalar values."""
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.state = None
        # Consecutive unmeasured samples. 0 means the last sample was good.
        self.gap = 0

    def update(self, x):
        """Fold `x` into the average and return the smoothed value.

        Returns NaN for an unmeasured sample without advancing the state, and
        None only while the filter has never seen a measurement.
        """
        if not is_measured(x):
            self.gap += 1
            return float("nan")
        self.gap = 0
        x = float(x)
        self.state = x if self.state is None else (
            self.alpha * x + (1 - self.alpha) * self.state)
        return self.state


class AngleSmoother:
    """
    Keeps an EMA per angle key.
    Usage:
        sm = AngleSmoother(alpha=0.25)
        smooth = sm(angles_dict)

    Unmeasured angles stay unmeasured. The previous implementation did
    `float(filter.update(v))`, which raised TypeError on the first sample of a
    key whose value was None, and turned one NaN into a permanently NaN filter.
    """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.filters = {}

    def _filter(self, key) -> EMA:
        if key not in self.filters:
            self.filters[key] = EMA(self.alpha)
        return self.filters[key]

    def __call__(self, angles: dict) -> dict:
        out = {}
        for k, v in angles.items():
            out[k] = self._filter(k).update(v)
        return out

    def last_good(self, key):
        """The most recent smoothed value for `key`, or None if never measured.

        For display only. Showing this in place of a current reading, without
        marking it stale, is how a frozen angle looks like a steady driver.
        """
        f = self.filters.get(key)
        return None if f is None else f.state

    def gap(self, key) -> int:
        """Consecutive unmeasured samples for `key`; 0 when the last was good."""
        f = self.filters.get(key)
        return 0 if f is None else f.gap
