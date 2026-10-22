"""Shared result types that every tracker returns, so overlays, filters and the sender read one format."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Keypoint:
    """One tracked point.

    x, y are normalized to 0-1 across the image (0, 0 = top-left), so they work at any size.
    z is MediaPipe's rough relative depth (smaller = closer to the camera).
    confidence is 0-1: how sure the tracker is that this point is visible and correct.
    """

    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0


@dataclass
class TrackResult:
    """Everything one tracker found in one frame.

    keypoints maps a unique name like "left_hand.wrist" to its Keypoint. A name that's
    missing means the tracker didn't see that point in this frame.
    """

    source: str        # which tracker made this, e.g. "hands" or "body"
    timestamp: float   # copied from the frame, so results can be matched to frames later
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
