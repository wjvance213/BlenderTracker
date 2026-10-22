"""Trackers: each one turns a camera frame into a TrackResult."""

from abc import ABC, abstractmethod

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from bodytracker.results import Keypoint, TrackResult

# MediaPipe's 21 hand landmarks, in the order the model returns them.
HAND_POINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# MediaPipe's 33 pose landmarks as (part, name), in the order the model returns them.
# Part None = pose's rough hand points, never used because HandTracker does hands better.
POSE_POINTS = (
    ("head", "nose"),
    ("head", "left_eye_inner"), ("head", "left_eye"), ("head", "left_eye_outer"),
    ("head", "right_eye_inner"), ("head", "right_eye"), ("head", "right_eye_outer"),
    ("head", "left_ear"), ("head", "right_ear"),
    ("head", "mouth_left"), ("head", "mouth_right"),
    ("shoulders", "left_shoulder"), ("shoulders", "right_shoulder"),
    ("arms", "left_elbow"), ("arms", "right_elbow"),
    ("arms", "left_wrist"), ("arms", "right_wrist"),
    (None, "left_pinky"), (None, "right_pinky"),
    (None, "left_index"), (None, "right_index"),
    (None, "left_thumb"), (None, "right_thumb"),
    ("hips", "left_hip"), ("hips", "right_hip"),
    ("legs", "left_knee"), ("legs", "right_knee"),
    ("legs", "left_ankle"), ("legs", "right_ankle"),
    ("feet", "left_heel"), ("feet", "right_heel"),
    ("feet", "left_foot_index"), ("feet", "right_foot_index"),
)
BODY_PARTS = ("head", "shoulders", "arms", "hips", "legs", "feet")


def _to_mp_image(frame):
    """Convert an OpenCV BGR frame into the RGB image type MediaPipe expects."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class Tracker(ABC):
    """The shape every tracker follows, so callers can run any of them the same way.

    Only pass *new* frames (a timestamp you haven't sent before): MediaPipe's video mode
    raises an error if a timestamp repeats or goes backwards.
    """

    @abstractmethod
    def process(self, frame, timestamp) -> TrackResult:
        """Find points in a BGR frame (as a stream gives it) and return them as a TrackResult."""

    def close(self):
        """Free the model's resources. Trackers without any can leave this as is."""


class HandTracker(Tracker):
    """Find hands with MediaPipe's HandLandmarker: 21 points per hand, named like "left_hand.wrist"."""

    def __init__(self, model_path, num_hands=2, swap_handedness=False):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode uses the previous frame to find hands faster, which is why
            # it needs timestamps that always increase.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.swap_handedness = swap_handedness

    def process(self, frame, timestamp):
        detection = self._landmarker.detect_for_video(_to_mp_image(frame), int(timestamp * 1000))

        result = TrackResult("hands", timestamp)
        # One entry per hand found: 21 landmarks, plus its best left/right guess and score.
        for landmarks, handedness in zip(detection.hand_landmarks, detection.handedness):
            best = handedness[0]
            side = best.category_name.lower()  # "left" or "right"
            if self.swap_handedness:
                side = "right" if side == "left" else "left"
            if f"{side}_hand.wrist" in result.keypoints:
                continue  # both hands got the same label; keep the first one
            for name, landmark in zip(HAND_POINT_NAMES, landmarks):
                result.keypoints[f"{side}_hand.{name}"] = Keypoint(landmark.x, landmark.y, landmark.z, best.score)
        return result

    def close(self):
        self._landmarker.close()


class BodyTracker(Tracker):
    """Find one person with MediaPipe's PoseLandmarker, keeping only the parts asked for.

    Points are named "<part>.<point>", like "arms.left_elbow". A point is left out when its
    part isn't in `parts` or the model's visibility for it is below `min_visibility`
    (usually because it's outside the camera's view and the model is only guessing).
    """

    def __init__(self, model_path, parts=("head", "shoulders", "arms"), min_visibility=0.5):
        unknown = set(parts) - set(BODY_PARTS)
        if unknown:
            raise ValueError(f"Unknown body parts {sorted(unknown)}. Choose from: {BODY_PARTS}")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.parts = set(parts)
        self.min_visibility = min_visibility

    def process(self, frame, timestamp):
        detection = self._landmarker.detect_for_video(_to_mp_image(frame), int(timestamp * 1000))

        result = TrackResult("body", timestamp)
        if not detection.pose_landmarks:
            return result  # nobody in view
        for (part, name), landmark in zip(POSE_POINTS, detection.pose_landmarks[0]):
            if part not in self.parts or landmark.visibility < self.min_visibility:
                continue
            result.keypoints[f"{part}.{name}"] = Keypoint(landmark.x, landmark.y, landmark.z, landmark.visibility)
        return result

    def close(self):
        self._landmarker.close()


def trackers_from_config(config):
    """Build the trackers switched on in config.toml (its [hands] and [body] sections)."""
    trackers = []
    hands = config["hands"]
    if hands["enabled"]:
        trackers.append(HandTracker(hands["model"], hands["num_hands"], hands["swap_handedness"]))
    body = config["body"]
    if body["enabled"]:
        trackers.append(BodyTracker(body["model"], body["parts"], body["min_visibility"]))
    return trackers


def _summary(result):
    """One short line per result: hand wrists, or every body point except the (many) head ones."""
    if result.source == "hands":
        shown = {key: point for key, point in result.keypoints.items() if key.endswith(".wrist")}
    else:
        shown = {key: point for key, point in result.keypoints.items() if not key.startswith("head.")}
        head_count = len(result.keypoints) - len(shown)
        if head_count:
            shown = {f"head ({head_count} pts)": None, **shown}
    parts = [key if point is None else f"{key} ({point.x:.2f}, {point.y:.2f})" for key, point in shown.items()]
    return "  ".join(parts) or "-"


if __name__ == "__main__":
    import sys
    import time
    import tomllib

    from bodytracker.camera import CameraStream, VideoFileStream

    # python -m bodytracker.trackers                        live camera
    # python -m bodytracker.trackers recordings\test.mp4    a video file instead
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    if len(sys.argv) > 1:
        stream = VideoFileStream(sys.argv[1])
    else:
        camera = config["camera"]
        stream = CameraStream(camera["name"], camera["width"], camera["height"], camera["fps"])
    trackers = trackers_from_config(config)
    if not trackers:
        sys.exit("Every tracker is disabled in config.toml.")

    stream.start()
    print("Tracking for 10 seconds...")
    last_timestamp = None
    last_print = 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < 10.0:
        frame, timestamp = stream.read()
        if timestamp is None or timestamp == last_timestamp:
            time.sleep(0.001)
            continue  # no new frame yet
        last_timestamp = timestamp

        lines = []
        for tracker in trackers:
            began = time.perf_counter()
            result = tracker.process(frame, timestamp)
            took_ms = (time.perf_counter() - began) * 1000
            lines.append(f"{took_ms:5.1f} ms  {result.source:5s} | {_summary(result)}")

        # Print twice a second, not every frame, so the output stays readable.
        if time.perf_counter() - last_print >= 0.5:
            last_print = time.perf_counter()
            print("\n".join(lines) + "\n")

    stream.stop()
    for tracker in trackers:
        tracker.close()
