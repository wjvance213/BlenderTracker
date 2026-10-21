# Body Tracker — working agreement

I'm building this project to learn. Understanding each step matters more than speed.

## Project
Track hands (MediaPipe Hands), head and feet (pose model) from a camera, smooth the
points with a One Euro filter, and send them over UDP to a Blender add-on that moves
empties/bones. Python CV process and Blender run as separate processes. Stereo (second
camera) comes last.

## Platform
Windows only. Don't add Linux/macOS branches, sys.platform checks, or platform markers.
Use Windows paths and commands (PowerShell, `.venv\Scripts\activate`) in instructions.

## Phases (do them in order; don't start a phase until the current one is done)
0. Repo + Python 3.11 venv + requirements.txt + .gitignore + config file
1. Camera discovery by name (pygrabber device list, opened with cv2.CAP_DSHOW so
   indices match), threaded CameraStream (latest frame + timestamp), VideoFileStream with the same
   interface, OpenCV tiled viewer with FPS
2. Shared types (Keypoint, TrackResult), tracker interface `process(frame, ts)`,
   HandTracker (MediaPipe Tasks HandLandmarker), BodyTracker (head + feet), overlays
3. One Euro filter per point per axis, reset after dropouts, tuning trackbars
4. Versioned JSON UDP sender + standalone listener (rate + seq gaps)
5. Blender add-on: non-blocking socket polled by bpy.app.timers, empties first,
   then bones via constraints
6. Stereo calibration + timestamp pairing + triangulation

## How to work with me
- Work on ONE small step at a time (one file or one function). Never implement a whole
  phase in one go.
- Before writing code: explain in a few sentences what the step does, why it's needed,
  and which approach you'll take. Mention any alternative worth knowing about. Wait for
  my OK.
- After writing code: walk through the important lines, especially any OpenCV,
  MediaPipe, threading, socket or bpy calls I may not know. Tell me exactly how to run
  or test it and what I should see.
- Stop after each step and wait for me. Don't move on because a step "looks done".
- Keep code simple and readable over clever. No new dependencies without asking.
- If something I ask for conflicts with this plan or is a bad idea, say so.
- When a step is verified, suggest a short commit message.
- When I ask "why", answer the question; don't rewrite code unless I ask.
