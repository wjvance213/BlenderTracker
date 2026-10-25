"""Stand-in for Blender: receive our UDP packets and print a status line once a second.

Run from the project folder (it reads config.toml there):

    python tools\\udp_listener.py

Only uses the standard library, like the Blender add-on will have to (Blender's Python
can't import bodytracker), so this is also a rehearsal for how Blender reads packets.
Stop with Ctrl+C.
"""

import json
import socket
import time
import tomllib

PACKET_VERSION = 1  # must match bodytracker/packet.py


def read_packet(data):
    """Packet bytes -> dict, or None if it isn't a packet we understand."""
    try:
        packet = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(packet, dict) or packet.get("v") != PACKET_VERSION:
        return None
    return packet


def main():
    with open("config.toml", "rb") as f:
        udp = tomllib.load(f)["udp"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # bind = claim this address + port, so packets sent to it come to us.
    sock.bind((udp["host"], udp["port"]))
    # Wake up every second even with no data: Windows ignores Ctrl+C while a socket waits forever.
    sock.settimeout(1.0)
    print(f"listening on {udp['host']}:{udp['port']} (Ctrl+C to stop)")

    last_seq = None
    lost = late = bad = 0
    count = 0            # packets since the last status line
    newest = None        # (number of points, size in bytes) of the newest packet
    window_start = time.perf_counter()

    try:
        while True:
            try:
                # 65535 = the biggest a UDP packet can be, so one recvfrom always gets a whole packet.
                data, _sender = sock.recvfrom(65535)
            except socket.timeout:
                data = None

            if data is not None:
                packet = read_packet(data)
                if packet is None:
                    bad += 1
                else:
                    seq = packet["seq"]
                    if seq == 0 and last_seq is not None:
                        print("sender restarted (seq 0), counts reset")
                        last_seq, lost, late, bad = None, 0, 0, 0
                    if last_seq is None or seq > last_seq:
                        if last_seq is not None:
                            lost += seq - last_seq - 1  # seqs we skipped over never arrived
                        last_seq = seq
                        count += 1
                        newest = (len(packet["points"]), len(data))
                    else:
                        late += 1  # older than one we already used: skip it, like Blender will

            now = time.perf_counter()
            if now - window_start >= 1.0:
                rate = count / (now - window_start)
                if newest is None:
                    print("waiting for packets...")
                else:
                    points, size = newest
                    print(f"{rate:5.1f} pkt/s | seq {last_seq} | lost {lost} | late {late} | bad {bad}"
                          f" | {points} points, {size} B")
                count = 0
                window_start = now
    except KeyboardInterrupt:
        print("stopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
