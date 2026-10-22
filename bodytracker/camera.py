"""Frame sources: camera discovery by DirectShow name, a live camera stream, and a video file stream."""

import threading
import time

import cv2
from pygrabber.dshow_graph import FilterGraph


def list_cameras():
    """Return camera names in DirectShow order (position = OpenCV index with CAP_DSHOW)."""
    return FilterGraph().get_input_devices()


def find_camera_index(name):
    """Return the index of the camera called `name` (case-insensitive exact match)."""
    names = list_cameras()
    for index, camera_name in enumerate(names):
        if camera_name.lower() == name.lower():
            return index
    raise ValueError(f"No camera named {name!r}. Found: {names}")


class _FrameStream:
    """Shared parts of every stream: a reader thread that keeps only the newest frame.

    Subclasses open their source in start(), then call _start_thread(), and implement
    _reader() to loop while self._running, handing each frame to _publish().
    """

    def __init__(self):
        self._cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = None

    def _start_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _publish(self, frame, timestamp):
        """Overwrite the newest frame (older ones are simply dropped)."""
        with self._lock:
            self._frame = frame
            self._timestamp = timestamp

    def read(self):
        """Return (frame, timestamp) for the newest frame, or (None, None) before the first one."""
        with self._lock:
            return self._frame, self._timestamp

    def stop(self):
        """Stop the thread and release the source."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
        if self._cap is not None:
            self._cap.release()


class CameraStream(_FrameStream):
    """Read a camera on a background thread and keep only the newest frame."""

    def __init__(self, name, width, height, fps):
        super().__init__()
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

    def start(self):
        """Open the camera, apply size/fps, and start the reader thread."""
        index = find_camera_index(self.name)
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.name!r} (index {index})")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # The camera may not give exactly what we asked for, so report what it chose.
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"{self.name}: {actual_width}x{actual_height} @ {actual_fps:g} fps")

        # Some cameras (like the Brio 101) only offer one size, so shrink frames ourselves.
        self._resize = (actual_width, actual_height) != (self.width, self.height)
        if self._resize:
            print(f"Resizing frames to {self.width}x{self.height}")

        self._start_thread()

    def _reader(self):
        """Thread loop: read frames as fast as they arrive, overwriting the last one."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)  # don't spin the CPU if the camera hiccups
                continue
            timestamp = time.perf_counter()
            if self._resize:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self._publish(frame, timestamp)


class VideoFileStream(_FrameStream):
    """Play a video file at its own fps on a background thread, like a live camera."""

    def __init__(self, path, loop=True):
        super().__init__()
        self.path = path
        self.loop = loop

    def start(self):
        """Open the file, read its fps, and start the playback thread."""
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file {self.path!r}")

        # Some files don't store an fps (OpenCV then returns 0), so fall back to 30.
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{self.path}: {width}x{height} @ {self.fps:g} fps, {frame_count} frames")

        self._start_thread()

    def _reader(self):
        """Thread loop: release frame i at start + i / fps, so playback runs in real time."""
        start = time.perf_counter()
        index = 0
        while self._running:
            # Sleep until this frame is due. Scheduling from `start` (not "sleep 33 ms
            # after each frame") keeps small timing errors from adding up.
            wait = start + index / self.fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

            ok, frame = self._cap.read()
            if not ok:
                if self.loop and index > 0:  # index 0 failing means the file has no frames
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start = time.perf_counter()
                    index = 0
                    continue
                break

            self._publish(frame, time.perf_counter())
            index += 1


if __name__ == "__main__":
    import sys
    import tomllib

    if len(sys.argv) > 1:
        # python -m bodytracker.camera recordings\test.mp4
        stream = VideoFileStream(sys.argv[1])
    else:
        for index, camera_name in enumerate(list_cameras()):
            print(f"{index}: {camera_name}")

        with open("config.toml", "rb") as f:
            camera = tomllib.load(f)["camera"]
        stream = CameraStream(camera["name"], camera["width"], camera["height"], camera["fps"])

    stream.start()

    # Count how many *new* frames arrive in 3 seconds (a new frame has a new timestamp).
    new_frames = 0
    last_timestamp = None
    start = time.perf_counter()
    while time.perf_counter() - start < 3.0:
        frame, timestamp = stream.read()
        if timestamp is not None and timestamp != last_timestamp:
            new_frames += 1
            last_timestamp = timestamp
        time.sleep(0.001)

    stream.stop()

    if last_timestamp is None:
        print("No frames arrived.")
    else:
        height, width = frame.shape[:2]
        print(f"Last frame {width}x{height}, about {new_frames / 3.0:.1f} new frames per second")
