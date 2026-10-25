"""Send each frame's points over UDP, one packet per frame (format in packet.py).

UDP is fire-and-forget: nothing waits for Blender to answer, and a lost packet is
simply gone. That's what we want, because a newer frame is only ~33 ms behind it.
See project_planning/udp_packets.html for why.
"""

import socket

from bodytracker.packet import make_packet


class UdpSender:
    """Owns one UDP socket and the packet counter (seq)."""

    def __init__(self, host, port):
        self.address = (host, port)
        # AF_INET = IPv4 addresses, SOCK_DGRAM = UDP (separate messages, no connection).
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, timestamp, results):
        """Send one frame's TrackResults as a packet. Returns the packet's size in bytes."""
        data = make_packet(self.seq, timestamp, results)
        # seq goes up even if this send fails, so the receiver sees the gap.
        self.seq += 1
        try:
            self.sock.sendto(data, self.address)
        except ConnectionResetError:
            # Windows can report "nobody was listening" for an earlier packet on a later send.
            # Blender being closed must never crash the tracker, so ignore it.
            pass
        return len(data)

    def close(self):
        self.sock.close()


def sender_from_config(config):
    """Build a UdpSender from config.toml's [udp] section."""
    udp = config["udp"]
    return UdpSender(udp["host"], udp["port"])


if __name__ == "__main__":
    import time
    import tomllib

    from bodytracker.results import Keypoint, TrackResult

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # Nothing listens yet (that's step 3), so this only shows sending works without crashing.
    sender = sender_from_config(config)
    print(f"sending 5 packets to {sender.address[0]}:{sender.address[1]}")
    for i in range(5):
        now = time.perf_counter()
        fake = TrackResult("hands", now, {"left_hand.wrist": Keypoint(0.5, 0.5 + i * 0.01, 0.0, 0.9)})
        size = sender.send(now, [fake])
        print(f"  seq {sender.seq - 1}: {size} bytes")
        time.sleep(1 / 30)
    sender.close()
    print("done")
