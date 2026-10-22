"""Tiled OpenCV viewer: every stream in one window, each tile labelled with its name and FPS."""

import time

import cv2
import numpy as np

from bodytracker.camera import CameraStream, VideoFileStream

WINDOW_NAME = "Body Tracker"
TILE_WIDTH = 640
TILE_HEIGHT = 360
COLUMNS = 2  # tiles per row before wrapping to a new row


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


def make_tile(frame, name, fps):
    """Return a TILE_WIDTH x TILE_HEIGHT image of `frame` with its name and fps drawn on top."""
    if frame is None:
        tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        _draw_label(tile, "waiting for frames...", (20, TILE_HEIGHT // 2))
    else:
        # resize() returns a new image, so drawing on the tile never touches the
        # stream's own frame (which the reader thread may hand out again).
        tile = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    _draw_label(tile, f"{name}  {fps:.1f} fps", (10, 25))
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


def show_streams(streams, names):
    """Show all streams in one window until q, Esc, or the window's X is pressed."""
    meters = [FpsMeter() for _ in streams]
    while True:
        tiles = []
        for stream, name, meter in zip(streams, names, meters):
            frame, timestamp = stream.read()
            meter.update(timestamp)
            tiles.append(make_tile(frame, name, meter.fps))

        cv2.imshow(WINDOW_NAME, tile_grid(tiles))

        # waitKey is what actually redraws the window and reads the keyboard.
        # It waits up to 1 ms and returns -1 if no key was pressed.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = Esc
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with its X button

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import tomllib
    from pathlib import Path

    # python -m bodytracker.viewer                        camera only
    # python -m bodytracker.viewer recordings\test.mp4    camera + one tile per video
    with open("config.toml", "rb") as f:
        camera = tomllib.load(f)["camera"]
    streams = [CameraStream(camera["name"], camera["width"], camera["height"], camera["fps"])]
    names = [camera["name"]]
    for path in sys.argv[1:]:
        streams.append(VideoFileStream(path))
        names.append(Path(path).name)

    for stream in streams:
        stream.start()
    try:
        show_streams(streams, names)
    finally:
        for stream in streams:
            stream.stop()
