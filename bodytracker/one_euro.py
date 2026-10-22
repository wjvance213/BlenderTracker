"""One Euro filter: smooths a noisy number heavily when it's still and lightly when it moves fast.

From Casiez, Roussel and Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy
Input in Interactive Systems" (CHI 2012).
"""

import math

from bodytracker.results import Keypoint, TrackResult


def _alpha(cutoff, dt):
    """How much of the new value to keep (0-1) for a low-pass filter at `cutoff` Hz.

    A higher cutoff, or a longer time since the last value, keeps more of the new value
    (= lighter smoothing).
    """
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Smooth one number over time. Use one filter per value you track (e.g. per point per axis).

    min_cutoff: smoothing when still, in Hz. Lower = smoother, but more lag on slow moves.
    beta: how quickly smoothing eases off as speed rises. Higher = less lag on fast moves.
    d_cutoff: smoothing of the speed estimate itself; rarely needs changing.
    """

    def __init__(self, min_cutoff=0.5, beta=4.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        """Forget the past, e.g. after a point drops out, so it doesn't glide in from its old spot."""
        self._value = None
        self._speed = 0.0
        self._timestamp = None

    def filter(self, value, timestamp):
        """Return the smoothed value. `timestamp` is in seconds (like the streams' perf_counter)."""
        if self._value is None:
            # First value: nothing to smooth against yet.
            self._value = value
            self._timestamp = timestamp
            return value

        # 1. Time step: fps varies, so measure it instead of assuming 1/30.
        dt = timestamp - self._timestamp
        if dt <= 0:
            return self._value  # same or older timestamp: nothing new to learn

        # 2. Speed (units per second), lightly smoothed so noise doesn't look like motion.
        speed = (value - self._value) / dt
        a_speed = _alpha(self.d_cutoff, dt)
        self._speed = a_speed * speed + (1 - a_speed) * self._speed

        # 3. Faster movement -> higher cutoff -> lighter smoothing.
        cutoff = self.min_cutoff + self.beta * abs(self._speed)

        # 4. Blend the new value with the last smoothed one.
        a = _alpha(cutoff, dt)
        self._value = a * value + (1 - a) * self._value
        self._timestamp = timestamp
        return self._value


class KeypointSmoother:
    """Smooth every point of one tracker's results over time: a OneEuroFilter per point per axis.

    Use one smoother per tracker per stream, since its filters remember that tracker's past points.
    """

    def __init__(self, min_cutoff=0.5, beta=4.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters = {}  # point name -> (x filter, y filter, z filter)

    def set_params(self, min_cutoff, beta):
        """Change the smoothing while running. Existing filters keep their history, so nothing jumps."""
        self.min_cutoff = min_cutoff
        self.beta = beta
        for filters in self._filters.values():
            for one_filter in filters:
                one_filter.min_cutoff = min_cutoff
                one_filter.beta = beta

    def smooth(self, result):
        """Return a new TrackResult with every point smoothed. `result` itself isn't changed."""
        # Points missing from this frame lose their filters, so when they come back they
        # start fresh where they reappear instead of gliding in from where they vanished.
        self._filters = {name: filters for name, filters in self._filters.items() if name in result.keypoints}

        smoothed = TrackResult(result.source, result.timestamp)
        for name, point in result.keypoints.items():
            if name not in self._filters:
                self._filters[name] = tuple(OneEuroFilter(self.min_cutoff, self.beta) for _ in range(3))
            filter_x, filter_y, filter_z = self._filters[name]
            t = result.timestamp
            smoothed.keypoints[name] = Keypoint(
                filter_x.filter(point.x, t),
                filter_y.filter(point.y, t),
                filter_z.filter(point.z, t),
                point.confidence,  # confidence isn't smoothed: it describes this frame
            )
        return smoothed


if __name__ == "__main__":
    import random
    import statistics

    # A fake wrist x at 30 fps, like the chart in project_planning/smoothing.html:
    # still at 0.3, a fast move to 0.7 (2.0-2.3 s), still, then a slow drift down.
    def true_x(t):
        if t < 2.0:
            return 0.3
        if t < 2.3:
            u = (t - 2.0) / 0.3
            return 0.3 + 0.4 * u * u * (3 - 2 * u)  # smooth S-curve
        if t < 4.0:
            return 0.7
        return 0.7 - 0.1 * (t - 4.0) / 2.0

    fps = 30
    times = [i / fps for i in range(6 * fps)]
    rng = random.Random(7)  # fixed seed, so every run prints the same numbers
    raw = [true_x(t) + rng.gauss(0, 0.006) for t in times]

    euro = OneEuroFilter()
    smoothed = [euro.filter(x, t) for x, t in zip(raw, times)]
    window = 8
    averaged = [statistics.fmean(raw[max(0, i - window + 1):i + 1]) for i in range(len(raw))]

    def wobble(values):
        """Spread of the values while holding still (0.5-2 s)."""
        return statistics.pstdev(v for v, t in zip(values, times) if 0.5 <= t < 2.0)

    def crossing_time(values):
        """When the line first reaches 0.66 (90% of the move), between frames."""
        for i in range(1, len(values)):
            if times[i] >= 2.0 and values[i - 1] < 0.66 <= values[i]:
                return times[i - 1] + (0.66 - values[i - 1]) / (values[i] - values[i - 1]) / fps
        return None

    true_cross = crossing_time([true_x(t) for t in times])
    print(f"{'':16s}{'wobble (still)':>16s}{'lag (fast move)':>18s}")
    for name, values in (("raw", raw), (f"average of {window}", averaged), ("One Euro", smoothed)):
        lag_ms = max(0.0, (crossing_time(values) - true_cross) * 1000)
        print(f"{name:16s}{wobble(values):16.4f}{lag_ms:15.0f} ms")
