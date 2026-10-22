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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect_for_video(image, int(timestamp * 1000))

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
    hands = config["hands"]
    tracker = HandTracker(hands["model"], hands["num_hands"], hands["swap_handedness"])

    stream.start()
    print("Tracking hands for 10 seconds...")
    last_timestamp = None
    last_print = 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < 10.0:
        frame, timestamp = stream.read()
        if timestamp is None or timestamp == last_timestamp:
            time.sleep(0.001)
            continue  # no new frame yet
        last_timestamp = timestamp

        began = time.perf_counter()
        result = tracker.process(frame, timestamp)
        took_ms = (time.perf_counter() - began) * 1000

        # Print twice a second, not every frame, so the output stays readable.
        if time.perf_counter() - last_print >= 0.5:
            last_print = time.perf_counter()
            wrists = [f"{key.split('.')[0]} ({point.x:.2f}, {point.y:.2f}, conf {point.confidence:.2f})"
                      for key, point in result.keypoints.items() if key.endswith(".wrist")]
            print(f"{took_ms:5.1f} ms | {len(wrists)} hand(s) | " + ("  ".join(wrists) or "-"))

    stream.stop()
    tracker.close()
