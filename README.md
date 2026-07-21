# Two-Hand Air Drawing (MediaPipe + OpenCV)

A real-time webcam application that lets you draw in the air using hand gestures. Your **right hand** draws, your **left hand** erases, and a **closed fist (either hand)** pans the entire canvas. Built with MediaPipe Hands for landmark tracking and OpenCV for rendering, with a soft glow effect applied to strokes.

## Features

- **Draw** — right hand, index finger extended only
- **Erase** — left hand, index + middle fingers extended (erases any stroke within a radius of the fingertip midpoint)
- **Pan / drag canvas** — closed fist on either hand, moves all existing strokes with the hand
- **Depth-reactive stroke thickness** — uses the index fingertip's estimated Z position (distance from camera) to vary line thickness
- **Glow effect** — strokes are blurred and blended back onto the frame for a neon-style look
- **Persistent, non-destructive strokes** — each stroke is stored as vector data (`p1`, `p2`, color, thickness) rather than burned into a pixel buffer, so panning doesn't degrade or clip the drawing
- Clear canvas and quit via keyboard shortcuts

## Requirements

- Python **3.9–3.11** (MediaPipe does not yet reliably support Python 3.12+; check the [MediaPipe PyPI page](https://pypi.org/project/mediapipe/) for current compatibility before installing)
- A working webcam
- A desktop environment with GUI support (this script uses `cv2.imshow`, so it will **not** run as-is on a headless server or inside most Docker containers/WSL without extra X11 setup)

## Installation

```bash
git clone https://github.com//jamalsatria13-svg/hand-tracking.git
cd <your-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python hand_tracking_2.py
```

| Gesture | Hand | Action |
|---|---|---|
| Index finger up only | Right | Draw |
| Index + middle fingers up | Left | Erase nearby strokes |
| Closed fist | Either | Pan/drag the whole canvas |

**Keyboard controls:**

| Key | Action |
|---|---|
| `c` | Clear the entire canvas |
| `q` | Quit the application |

## How it works

1. Each frame is flipped horizontally (mirror view) and passed to MediaPipe Hands, which returns up to 2 hands with 21 landmarks each plus a Left/Right label.
2. A simple heuristic (`fingertip.y < pip_joint.y`) determines which fingers are extended.
3. Depending on the detected hand and gesture, the app either appends a new line segment to the stroke list, removes nearby strokes, or applies a pan offset to every stored stroke.
4. All strokes are redrawn from scratch every frame onto a blank canvas, blurred for the glow layer, and composited back onto the live camera feed.

## Known limitations

- The finger-extension heuristic is orientation-sensitive — it can misfire if the hand is rotated, tilted, or facing away from the camera.
- Depth (Z) values from MediaPipe are relative and uncalibrated, not true distance, so the thickness effect is approximate and may vary by camera/lighting.
- Only two hands are tracked at once (`max_num_hands=2`), and the left/right roles are fixed — there's no swap option.
- No save/export/undo functionality; closing the app discards the drawing.

## THANK YOU...
