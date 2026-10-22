"""Draw TrackResults on images: a dot per keypoint and a line per bone."""

import cv2

from bodytracker.trackers import HAND_POINT_NAMES

# OpenCV colours are BGR, not RGB.
LEFT_COLOR = (43, 130, 217)     # orange: your left side
RIGHT_COLOR = (216, 111, 59)    # blue: your right side
MIDDLE_COLOR = (170, 170, 170)  # grey: head, and lines that cross the middle

# Which hand points are joined, as index pairs into HAND_POINT_NAMES (MediaPipe's layout).
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky and palm edge
)


def _hand_bones():
    bones = []
    for side in ("left", "right"):
        for a, b in _HAND_CONNECTIONS:
            bones.append((f"{side}_hand.{HAND_POINT_NAMES[a]}", f"{side}_hand.{HAND_POINT_NAMES[b]}"))
    return bones


def _body_bones():
    bones = [
        ("shoulders.left_shoulder", "shoulders.right_shoulder"),
        ("hips.left_hip", "hips.right_hip"),
    ]
    for side in ("left", "right"):
        bones += [
            (f"shoulders.{side}_shoulder", f"arms.{side}_elbow"),
            (f"arms.{side}_elbow", f"arms.{side}_wrist"),
            (f"shoulders.{side}_shoulder", f"hips.{side}_hip"),
            (f"hips.{side}_hip", f"legs.{side}_knee"),
            (f"legs.{side}_knee", f"legs.{side}_ankle"),
            (f"legs.{side}_ankle", f"feet.{side}_heel"),
            (f"feet.{side}_heel", f"feet.{side}_foot_index"),
            (f"legs.{side}_ankle", f"feet.{side}_foot_index"),
        ]
    return bones


# Bones for each tracker, looked up by TrackResult.source.
BONES = {"hands": _hand_bones(), "body": _body_bones()}


def _color(key):
    """Colour a point by which side of the body its name says it's on (the head stays grey)."""
    if key.startswith("head."):
        return MIDDLE_COLOR
    if "left" in key:
        return LEFT_COLOR
    if "right" in key:
        return RIGHT_COLOR
    return MIDDLE_COLOR


def draw_result(image, result):
    """Draw one TrackResult onto `image` (changes it in place). Works at any image size."""
    height, width = image.shape[:2]
    # Keypoints are 0-1, so multiplying by the image size gives pixels.
    pixels = {key: (int(point.x * width), int(point.y * height)) for key, point in result.keypoints.items()}

    # Lines first, so the dots sit on top of them.
    for a, b in BONES.get(result.source, ()):
        if a in pixels and b in pixels:  # skip a bone if either end wasn't seen
            color = _color(a) if _color(a) == _color(b) else MIDDLE_COLOR
            cv2.line(image, pixels[a], pixels[b], color, 2, cv2.LINE_AA)

    for key, pixel in pixels.items():
        radius = 2 if key.startswith("head.") else 3 if result.source == "hands" else 5
        cv2.circle(image, pixel, radius, _color(key), cv2.FILLED, cv2.LINE_AA)
