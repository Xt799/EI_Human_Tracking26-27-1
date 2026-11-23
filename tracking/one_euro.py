"""One Euro filter for a person's floor position: smooths a lot while they stand, little while they walk."""
import numpy as np

MIN_CUTOFF_HZ = 0.5      # smoothing while standing: lower is calmer
SPEED_GAIN = 2.0         # how quickly the smoothing lets go as the person moves, per m/s
SPEED_CUTOFF_HZ = 1.0    # smoothing of the speed estimate itself


def cutoff_weight(cutoff_hz, dt):
    """Weight of a new value in a low-pass filter with this cutoff frequency, dt seconds after the last one."""
    return 1 / (1 + 1 / (2 * np.pi * cutoff_hz * dt))


class OneEuro:
    """One Euro filter of a floor position: the faster the position moves, the less it is smoothed."""

    def __init__(self):
        self.x = None
        self.speed = np.zeros(2)     # smoothed velocity, m/s
        self.t = None

    def __call__(self, x, t):
        """The filtered position for a new measurement x at time t."""
        if self.x is None or t <= self.t:
            self.x, self.t = x, t
            return x
        dt = t - self.t
        w = cutoff_weight(SPEED_CUTOFF_HZ, dt)
        self.speed = w * (x - self.x) / dt + (1 - w) * self.speed
        w = cutoff_weight(MIN_CUTOFF_HZ + SPEED_GAIN * float(np.linalg.norm(self.speed)), dt)
        self.x, self.t = w * x + (1 - w) * self.x, t
        return self.x
