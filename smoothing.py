class EMA:
    """Exponential moving average for scalar values."""
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.state = None

    def update(self, x):
        if x is None:
            return self.state
        self.state = x if self.state is None else (self.alpha * x + (1 - self.alpha) * self.state)
        return self.state

class AngleSmoother:
    """
    Keeps an EMA per angle key.
    Usage:
        sm = AngleSmoother(alpha=0.25)
        smooth = sm(angles_dict)
    """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.filters = {}

    def __call__(self, angles: dict) -> dict:
        out = {}
        for k, v in angles.items():
            if k not in self.filters:
                self.filters[k] = EMA(self.alpha)
            out[k] = float(self.filters[k].update(v))
        return out
