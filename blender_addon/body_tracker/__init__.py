"""Body Tracker add-on: receives the tracker's UDP packets and moves objects in Blender.

Self-contained on purpose: Blender's Python has only the standard library and bpy, so
nothing from the bodytracker package can be imported here.

Start listens for packets (the panel shows packet rate, seq and lost/late/bad counts, like
tools/udp_listener.py) and every point in the newest packet moves an empty named
"BT <point name>" in the "BodyTracker" collection. Points missing from a packet (or below
the minimum confidence) are hidden or frozen. Open it with N in the 3D viewport,
tab "Body Tracker".
"""

import json
import socket
import time

import bpy

PACKET_VERSION = 1    # must match bodytracker/packet.py
POLL_INTERVAL = 1 / 60  # seconds between socket checks: twice the camera's 30 fps, so little waiting

COLLECTION_NAME = "BodyTracker"
NAME_PREFIX = "BT "
ASPECT = 9 / 16       # image height / width: y covers fewer pixels than x (1280x720)
HAND_SIZE = 0.01      # empty sphere radius in metres; small so fingers don't merge
BODY_SIZE = 0.03


def read_packet(data):
    """Packet bytes -> dict, or None if it isn't a packet we understand. Same as udp_listener.py."""
    try:
        packet = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(packet, dict) or packet.get("v") != PACKET_VERSION:
        return None
    return packet


class Receiver:
    """A non-blocking UDP socket plus the counts the panel shows."""

    def __init__(self, host, port):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(self.address)
        except OSError:
            self.sock.close()
            raise
        # Non-blocking: recvfrom returns straight away (raising BlockingIOError if nothing is
        # waiting) instead of freezing Blender until a packet arrives.
        self.sock.setblocking(False)
        self.packet = None   # the newest packet
        self.new_packet = False  # True when self.packet hasn't been applied to the scene yet
        self.last_seq = None
        self.lost = self.late = self.bad = 0
        self.rate = 0.0
        self._count = 0
        self._window_start = time.perf_counter()

    def poll(self):
        """Read every waiting packet. Returns True once a second, when the panel should redraw."""
        while True:
            try:
                data, _sender = self.sock.recvfrom(65535)
            except BlockingIOError:
                break  # nothing left to read
            except ConnectionResetError:
                continue  # Windows quirk, same as in sender.py: ignore it
            self._handle(data)

        now = time.perf_counter()
        if now - self._window_start >= 1.0:
            self.rate = self._count / (now - self._window_start)
            self._count = 0
            self._window_start = now
            return True
        return False

    def _handle(self, data):
        """Count one packet and keep it if it's the newest. Same rules as udp_listener.py."""
        packet = read_packet(data)
        if packet is None:
            self.bad += 1
            return
        seq = packet["seq"]
        if seq == 0 and self.last_seq is not None:
            # The tracker restarted, so its counts start over.
            self.last_seq, self.lost, self.late, self.bad = None, 0, 0, 0
        if self.last_seq is None or seq > self.last_seq:
            if self.last_seq is not None:
                self.lost += seq - self.last_seq - 1
            self.last_seq = seq
            self._count += 1
            self.packet = packet
            self.new_packet = True
        else:
            self.late += 1

    def close(self):
        self.sock.close()


# The running receiver, or None when stopped. A module variable, not a Scene property,
# because Blender can't save a socket into a .blend file.
_receiver = None


def _redraw_panels():
    """Blender only redraws a panel when something happens there, so ask for it."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def to_blender(x, y, z, scale):
    """Image coordinates (0-1, y down, z = depth) -> Blender location (Z up, -Y toward the viewer).

    Your left hand is on the image's right (bigger x), so it lands at +X: like a character
    facing you in Front view, which is how Blender rigs are set up. scale = Blender metres
    for the image's full width.
    """
    return ((x - 0.5) * scale,
            z * scale,                       # smaller z = closer to the camera = toward you
            (0.5 - y) * scale * ASPECT)      # flip y so up is up


def _get_collection():
    """The BodyTracker collection, created and linked to the scene if it's missing."""
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION_NAME)
    scene_root = bpy.context.scene.collection
    if scene_root.children.get(collection.name) is None:
        scene_root.children.link(collection)
    return collection


def _apply_packet(packet, scene):
    """Move one empty per point in the packet, creating any that don't exist yet.

    Empties whose point is missing (or below the minimum confidence) are hidden or left
    where they were, depending on the panel's Missing points setting.
    """
    collection = _get_collection()
    seen = set()
    for name, (x, y, z, confidence) in packet["points"].items():
        if confidence < scene.bodytracker_min_confidence:
            continue
        # Look the empty up by name every time instead of keeping a reference: after
        # undo or a delete, an old reference points at nothing, while a lookup just
        # finds nothing and makes a new one.
        object_name = NAME_PREFIX + name
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            obj = bpy.data.objects.new(object_name, None)  # None = no mesh data, so an empty
            obj.empty_display_type = "SPHERE"
            obj.empty_display_size = HAND_SIZE if name.split(".")[0].endswith("_hand") else BODY_SIZE
            collection.objects.link(obj)
        obj.location = to_blender(x, y, z, scene.bodytracker_scale)
        _set_hidden(obj, False)
        seen.add(object_name)

    if scene.bodytracker_missing == "HIDE":
        for obj in collection.objects:
            if obj.name.startswith(NAME_PREFIX) and obj.name not in seen:
                _set_hidden(obj, True)


def _set_hidden(obj, hidden):
    """Hide or show an empty (the Outliner's monitor icon). The empty keeps its last location."""
    # Only write when it changes: every write makes Blender re-evaluate the scene.
    if obj.hide_viewport != hidden:
        obj.hide_viewport = hidden


def _poll():
    """bpy.app.timers calls this on Blender's main thread, where touching bpy is safe."""
    if _receiver is None:
        return None  # returning None stops the timer
    stats_changed = _receiver.poll()
    # The timer runs at 60 Hz but packets come at ~30, so only move things for a new one.
    if _receiver.new_packet:
        _receiver.new_packet = False
        _apply_packet(_receiver.packet, bpy.context.scene)
    if stats_changed:
        _redraw_panels()
    return POLL_INTERVAL  # returning a number = "call me again in this many seconds"


def _stop_receiving():
    global _receiver
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    if _receiver is not None:
        _receiver.close()
        _receiver = None


class BODYTRACKER_OT_start(bpy.types.Operator):
    """Start receiving tracking points"""  # Blender shows this docstring as the button's tooltip

    # bl_idname is how Blender finds the operator: "<category>.<name>", lowercase.
    bl_idname = "bodytracker.start"
    bl_label = "Start"

    @classmethod
    def poll(cls, context):
        # Blender greys the button out while this returns False.
        return _receiver is None

    def execute(self, context):
        global _receiver
        scene = context.scene
        host, port = scene.bodytracker_host, scene.bodytracker_port
        try:
            _receiver = Receiver(host, port)
        except OSError as error:
            # Usually the port is taken: only one program can listen on it at a time.
            self.report({"ERROR"}, f"Can't listen on {host}:{port} (is tools\\udp_listener.py running?): {error}")
            return {"CANCELLED"}
        # persistent=True keeps the timer running when another .blend file is opened.
        bpy.app.timers.register(_poll, first_interval=0, persistent=True)
        self.report({"INFO"}, f"Listening on {host}:{port}")
        return {"FINISHED"}


class BODYTRACKER_OT_stop(bpy.types.Operator):
    """Stop receiving tracking points"""

    bl_idname = "bodytracker.stop"
    bl_label = "Stop"

    @classmethod
    def poll(cls, context):
        return _receiver is not None

    def execute(self, context):
        _stop_receiving()
        self.report({"INFO"}, "Stopped")
        return {"FINISHED"}


class BODYTRACKER_PT_panel(bpy.types.Panel):
    """The Body Tracker tab in the 3D viewport's sidebar (press N)."""

    bl_label = "Body Tracker"
    bl_space_type = "VIEW_3D"  # which editor
    bl_region_type = "UI"      # "UI" = the N sidebar
    bl_category = "Body Tracker"  # the tab's name

    def draw(self, context):
        # draw() runs every time Blender redraws the panel, so it only lays things out.
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "bodytracker_host", text="Host")
        layout.prop(scene, "bodytracker_port", text="Port")
        row = layout.row(align=True)
        row.operator(BODYTRACKER_OT_start.bl_idname, icon="PLAY")
        row.operator(BODYTRACKER_OT_stop.bl_idname, icon="SNAP_FACE")

        layout.separator()
        layout.label(text="Points")
        layout.prop(scene, "bodytracker_scale")
        layout.prop(scene, "bodytracker_min_confidence")
        # expand=True draws the choices as side-by-side toggle buttons instead of a dropdown.
        layout.row().prop(scene, "bodytracker_missing", expand=True)
        layout.separator()

        if _receiver is None:
            layout.label(text="Stopped", icon="RADIOBUT_OFF")
            return
        host, port = _receiver.address
        layout.label(text=f"Listening on {host}:{port}", icon="RADIOBUT_ON")
        if _receiver.packet is None:
            layout.label(text="waiting for packets...")
            return
        col = layout.column(align=True)
        col.label(text=f"{_receiver.rate:.1f} pkt/s  |  seq {_receiver.last_seq}")
        col.label(text=f"lost {_receiver.lost}  |  late {_receiver.late}  |  bad {_receiver.bad}")
        col.label(text=f"{len(_receiver.packet['points'])} points")


CLASSES = (BODYTRACKER_OT_start, BODYTRACKER_OT_stop, BODYTRACKER_PT_panel)


def register():
    """Blender calls this when the add-on is enabled."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    # Settings live on the Scene, so they save with the .blend file and the panel can edit them.
    bpy.types.Scene.bodytracker_host = bpy.props.StringProperty(
        name="Host", default="127.0.0.1", description="Address to listen on (127.0.0.1 = this PC)")
    bpy.types.Scene.bodytracker_port = bpy.props.IntProperty(
        name="Port", default=9100, min=1, max=65535, description="Must match [udp] port in config.toml")
    bpy.types.Scene.bodytracker_scale = bpy.props.FloatProperty(
        name="Scale", default=2.0, min=0.1, soft_max=10.0, unit="LENGTH",
        description="Blender size of the camera image's full width")
    bpy.types.Scene.bodytracker_min_confidence = bpy.props.FloatProperty(
        name="Min confidence", default=0.5, min=0.0, max=1.0,
        description="Treat points less certain than this as missing")
    # Each item is (identifier, label shown, tooltip).
    bpy.types.Scene.bodytracker_missing = bpy.props.EnumProperty(
        name="Missing points", default="HIDE",
        items=[("HIDE", "Hide", "Hide a point's empty until the point comes back"),
               ("FREEZE", "Freeze", "Leave a point's empty where it was last seen")])


def unregister():
    """Blender calls this when the add-on is disabled. Undo everything register() did."""
    # Stop first, or the timer would keep running (and the port stay taken) after disabling.
    _stop_receiving()
    del bpy.types.Scene.bodytracker_missing
    del bpy.types.Scene.bodytracker_min_confidence
    del bpy.types.Scene.bodytracker_scale
    del bpy.types.Scene.bodytracker_port
    del bpy.types.Scene.bodytracker_host
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
