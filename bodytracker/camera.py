"""Camera discovery: find an OpenCV camera index from a DirectShow device name."""

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


if __name__ == "__main__":
    import tomllib

    import cv2

    for index, camera_name in enumerate(list_cameras()):
        print(f"{index}: {camera_name}")

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    name = config["camera"]["name"]
    index = find_camera_index(name)

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    cap.release()

    if ok:
        height, width = frame.shape[:2]
        print(f"{name} -> index {index}, got frame {width}x{height}")
    else:
        print(f"{name} -> index {index}, but no frame could be read")
