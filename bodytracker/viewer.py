"""Tiled OpenCV viewer: every stream in one window, each tile labelled with its name and FPS,
with tracking results drawn on top."""

import time

import cv2
import numpy as np

from bodytracker.camera import CameraStream, VideoFileStream
from bodytracker.one_euro import KeypointSmoother
from bodytracker.overlay import draw_result
from bodytracker.sender import sender_from_config
from bodytracker.trackers import trackers_from_config

WINDOW_NAME = "Body Tracker"
TILE_WIDTH = 640
TILE_HEIGHT = 360
COLUMNS = 2  # tiles per row before wrapping to a new row

# Trackbars only hold whole numbers, so they store min_cutoff x100 and beta x10.
TRACKBAR_MIN_CUTOFF = "min_cutoff x100"
TRACKBAR_BETA = "beta x10"


class FpsMeter:
    """Count new frames and turn the count into frames per second every `interval` seconds."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.fps = 0.0
        self._count = 0
        self._last_timestamp = None
        self._window_start = time.perf_counter()

    def update(self, timestamp):
        """Call once per viewer loop with the stream's latest timestamp."""
        # The viewer loop runs faster than the camera, so it often sees the same frame
        # twice. A frame only counts as new if its timestamp changed.
        if timestamp is not None and timestamp != self._last_timestamp:
            self._count += 1
            self._last_timestamp = timestamp

        now = time.perf_counter()
        elapsed = now - self._window_start
        if elapsed >= self.interval:
            self.fps = self._count / elapsed
            self._count = 0
            self._window_start = now


def _draw_label(image, text, position):
    """White text on a black box, readable on any background. `position` is the text's bottom-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, 0.6, 1)
    x, y = position
    cv2.rectangle(image, (x - 4, y - height - 4), (x + width + 4, y + baseline + 2), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, position, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def make_tile(frame, name, fps, results=(), track_ms=None, smooth_on=None, sent_seq=None):
    """Return a TILE_WIDTH x TILE_HEIGHT image of `frame` with tracking results, name and fps on top."""
    if frame is None:
        tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        _draw_label(tile, "waiting for frames...", (20, TILE_HEIGHT // 2))
    else:
        # resize() returns a new image, so drawing on the tile never touches the
        # stream's own frame (which the reader thread may hand out again).
        tile = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        for result in results:
            draw_result(tile, result)
    label = f"{name}  {fps:.1f} fps"
    if track_ms is not None:
        label += f"  track {track_ms:.0f} ms"
    if smooth_on is not None:
        label += "  [s] smooth" if smooth_on else "  [s] RAW"
    if sent_seq is not None:
        label += f"  sent #{sent_seq}"
    _draw_label(tile, label, (10, 25))
    return tile


def tile_grid(tiles):
    """Join tiles into rows of COLUMNS, padding the last row with black so rows line up."""
    blank = np.zeros_like(tiles[0])
    rows = []
    for i in range(0, len(tiles), COLUMNS):
        row = tiles[i:i + COLUMNS]
        if len(tiles) > COLUMNS:
            row += [blank] * (COLUMNS - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def show_streams(streams, names, trackers_per_stream=None, smoothing=None, sender=None):
    """Show all streams in one window until q, Esc, or the window's X is pressed.

    trackers_per_stream holds one list of trackers for each stream (each stream needs its
    own, because MediaPipe's video mode remembers the previous frame it was given).
    smoothing is the config's [smoothing] section; s switches between smoothed and raw.
    sender (a UdpSender) gets the first stream's results for every new frame, smoothed or
    raw to match the screen. Only one stream sends, so Blender never gets two bodies mixed up.
    """
    if trackers_per_stream is None:
        trackers_per_stream = [[] for _ in streams]
    if smoothing is None:
        smoothing = {"enabled": False, "min_cutoff": 0.5, "beta": 4.0}
    smooth_on = smoothing["enabled"]
    # One smoother per tracker per stream, for the same reason as the trackers.
    smoothers = [[KeypointSmoother(smoothing["min_cutoff"], smoothing["beta"]) for _ in trackers]
                 for trackers in trackers_per_stream]
    meters = [FpsMeter() for _ in streams]
    # Per stream: the timestamp last tracked, its raw and smoothed results, and how long tracking took.
    last_tracked = [None for _ in streams]
    raw_results = [[] for _ in streams]
    smooth_results = [[] for _ in streams]
    track_ms = [None for _ in streams]

    # Trackbars (sliders) have to belong to a window, so create the window up front.
    cv2.namedWindow(WINDOW_NAME)
    has_trackers = any(trackers_per_stream)
    params = (smoothing["min_cutoff"], smoothing["beta"])
    if has_trackers:
        # The last argument is a callback for every change; we read the sliders in the loop instead.
        cv2.createTrackbar(TRACKBAR_MIN_CUTOFF, WINDOW_NAME, round(params[0] * 100), 500, lambda value: None)
        cv2.createTrackbar(TRACKBAR_BETA, WINDOW_NAME, round(params[1] * 10), 400, lambda value: None)
        print("Drag the sliders to tune smoothing; copy values you like into config.toml [smoothing].")

    while True:
        tiles = []
        for i, (stream, name, trackers) in enumerate(zip(streams, names, trackers_per_stream)):
            frame, timestamp = stream.read()
            meters[i].update(timestamp)

            # Track only new frames: MediaPipe rejects a repeated timestamp, and an old
            # frame's results are still the right ones to draw on it.
            if trackers and timestamp is not None and timestamp != last_tracked[i]:
                last_tracked[i] = timestamp
                began = time.perf_counter()
                raw_results[i] = [tracker.process(frame, timestamp) for tracker in trackers]
                track_ms[i] = (time.perf_counter() - began) * 1000
                # Smooth every frame, even while showing raw, so pressing s never makes
                # the filters jump to catch up.
                smooth_results[i] = [smoother.smooth(result) for smoother, result in zip(smoothers[i], raw_results[i])]
                if sender is not None and i == 0:
                    sender.send(timestamp, smooth_results[i] if smooth_on else raw_results[i])

            shown = smooth_results[i] if smooth_on else raw_results[i]
            # sender.seq is the next number to use, so the last one sent is seq - 1.
            sent_seq = sender.seq - 1 if sender is not None and i == 0 and sender.seq > 0 else None
            tiles.append(make_tile(frame, name, meters[i].fps, shown, track_ms[i], smooth_on if trackers else None,
                                   sent_seq))

        cv2.imshow(WINDOW_NAME, tile_grid(tiles))

        # waitKey is what actually redraws the window and reads the keyboard.
        # It waits up to 1 ms and returns -1 if no key was pressed.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = Esc
            break
        if key == ord("s"):
            smooth_on = not smooth_on
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with its X button

        if has_trackers:
            # A min_cutoff of 0 would divide by zero inside the filter, so the slider's floor is 0.01.
            new_params = (max(1, cv2.getTrackbarPos(TRACKBAR_MIN_CUTOFF, WINDOW_NAME)) / 100,
                          cv2.getTrackbarPos(TRACKBAR_BETA, WINDOW_NAME) / 10)
            if new_params != params:
                params = new_params
                for stream_smoothers in smoothers:
                    for smoother in stream_smoothers:
                        smoother.set_params(*params)
                print(f"min_cutoff = {params[0]:.2f}   beta = {params[1]:.1f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import tomllib
    from pathlib import Path

    # python -m bodytracker.viewer                        camera only
    # python -m bodytracker.viewer recordings\test.mp4    camera + one tile per video
    # Which trackers run (and which body parts) is set in config.toml.
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    camera = config["camera"]
    streams = [CameraStream(camera["name"], camera["width"], camera["height"], camera["fps"])]
    names = [camera["name"]]
    for path in sys.argv[1:]:
        streams.append(VideoFileStream(path))
        names.append(Path(path).name)
    trackers_per_stream = [trackers_from_config(config) for _ in streams]
    sender = sender_from_config(config)
    print(f"Sending the camera's points to {sender.address[0]}:{sender.address[1]}")

    for stream in streams:
        stream.start()
    try:
        show_streams(streams, names, trackers_per_stream, config["smoothing"], sender)
    finally:
        sender.close()
        for stream in streams:
            stream.stop()
        for trackers in trackers_per_stream:
            for tracker in trackers:
                tracker.close()
