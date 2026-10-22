# Body Tracker

Tracks hands and the body parts you choose from a webcam with MediaPipe, smooths the points with a
One Euro filter, and streams them over UDP to a Blender add-on that moves empties/bones.

```
camera -> trackers -> One Euro filter -> UDP (JSON) -> Blender add-on
```

Windows only. Python 3.13.

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r docs\requirements.txt
```

If `activate` is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Configure

Edit `config.toml`: camera name, resolution, fps, UDP host/port.

## Layout

| Path | What |
|---|---|
| `bodytracker/` | Python CV process (run modules with `python -m bodytracker.<module>`) |
| `blender_addon/` | Blender add-on (self-contained, uses Blender's own Python) |
| `tools/` | Standalone helpers, e.g. UDP listener |
| `config.toml` | Settings |
| `docs/requirements.txt` | Pinned Python dependencies |
