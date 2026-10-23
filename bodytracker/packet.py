"""The UDP packet format: one small JSON message per frame with every tracked point in it.

Example (pretty-printed; the real packet has no spaces):

    {"v": 1, "seq": 42, "time": 12.345,
     "points": {"left_hand.wrist": [0.4123, 0.5871, -0.0312, 0.97],
                "arms.left_elbow": [0.6801, 0.7012, -0.1544, 0.88]}}

v       format version. Bump it whenever the layout changes, so an old receiver can
        refuse a packet it doesn't understand instead of misreading it.
seq     packet counter, +1 per packet. A gap tells the receiver packets were lost.
time    the frame's timestamp in seconds (perf_counter, so only meaningful on this machine).
points  name -> [x, y, z, confidence]. Coordinates are the same as Keypoint: x, y are 0-1
        from the top-left of the image (y points DOWN; the receiver flips it if it needs to).
        A missing name means that point wasn't seen in this frame.
"""

import json

from bodytracker.results import Keypoint

PACKET_VERSION = 1
DECIMALS = 4  # 0.0001 of a 1280 px wide image is about 0.1 px, plenty, and keeps packets small


def make_packet(seq, timestamp, results):
    """Turn one frame's TrackResults into packet bytes ready to send."""
    points = {}
    for result in results:
        for name, point in result.keypoints.items():
            points[name] = [round(value, DECIMALS) for value in (point.x, point.y, point.z, point.confidence)]
    packet = {"v": PACKET_VERSION, "seq": seq, "time": round(timestamp, 6), "points": points}
    # separators without spaces: about 15% smaller than json.dumps' default.
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def parse_packet(data):
    """Turn packet bytes back into (seq, time, {name: Keypoint}). Raises ValueError if it isn't one."""
    try:
        packet = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"not a JSON packet: {error}") from error
    if not isinstance(packet, dict) or packet.get("v") != PACKET_VERSION:
        raise ValueError(f"expected packet version {PACKET_VERSION}, got {packet.get('v') if isinstance(packet, dict) else packet!r}")
    points = {name: Keypoint(*values) for name, values in packet["points"].items()}
    return packet["seq"], packet["time"], points


if __name__ == "__main__":
    import random

    from bodytracker.results import TrackResult
    from bodytracker.trackers import HAND_POINT_NAMES

    # A worst-case frame: both hands (21 points each) plus every body point the parts can give.
    rng = random.Random(1)

    def fake_point():
        return Keypoint(rng.random(), rng.random(), rng.uniform(-0.3, 0.3), rng.random())

    hands = TrackResult("hands", 12.345678, {f"{side}.{name}": fake_point()
                                             for side in ("left_hand", "right_hand") for name in HAND_POINT_NAMES})
    body = TrackResult("body", 12.345678, {f"arms.{name}": fake_point()
                                           for name in ("left_elbow", "right_elbow", "left_wrist", "right_wrist")})

    data = make_packet(42, 12.345678, [hands, body])
    print(f"{len(hands.keypoints) + len(body.keypoints)} points -> {len(data)} bytes")
    print(data[:150].decode() + " ...")

    seq, t, points = parse_packet(data)
    worst = max(abs(getattr(points[name], axis) - getattr(original, axis))
                for result in (hands, body) for name, original in result.keypoints.items()
                for axis in ("x", "y", "z", "confidence"))
    print(f"round trip: seq={seq} time={t} points={len(points)} biggest rounding error={worst:.6f}")

    for bad in (b"hello", b'{"v": 2, "seq": 0, "time": 0, "points": {}}'):
        try:
            parse_packet(bad)
        except ValueError as error:
            print(f"rejected {bad[:20]!r}: {error}")
