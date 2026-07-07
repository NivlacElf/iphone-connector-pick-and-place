# Autonomous iPhone Connector Pick-and-Place

A machine that picks up an Apple iPhone flex-cable plug and inserts it into its board socket — fully autonomously — built from a modified Ender 3 3D printer, two cameras, and three custom-trained YOLOv8 vision models.

Connector mating is one of the last steps in phone assembly still done by human hands: the plug is a few millimeters wide and the socket tolerance is fractions of a millimeter. This project closes that gap with closed-loop visual servoing instead of expensive precision robotics.

<!-- IMAGE: hero shot of the full rig (Ender 3 + toolhead + phone case holder), ~1200px wide -->
![Full rig](docs/media/hero.jpg)

**[▶ Watch the full demo video](YOUTUBE_LINK_HERE)**

## How It Works

The Ender 3 is used as a cheap, precise 3-axis CNC motion platform, driven over USB with G-code. Two cameras give the machine eyes:

| Camera | Direction | Job |
|---|---|---|
| Toolhead camera | Looking down | Finds the plug on the table, then finds the socket on the phone |
| Microscope | Looking up | Inspects the plug held by the picker to measure its exact offset and angle |

Three YOLOv8 oriented-bounding-box models handle detection: `plug_front.pt` (plug on the table), `plug_back.pt` (pin side of the plug, viewed from below), and `SocketML.pt` (board socket).

Every alignment uses the same closed-loop pattern: detect the target, convert its pixel offset from image center to millimeters (calibrated per Z height), move the printer, and re-detect — repeating until the error converges to near zero. No move is trusted blindly.

### The Pipeline

<!-- IMAGE: 4-panel strip showing phases 1, 3, 4, 5 (annotated camera views), or one diagram -->
![Pipeline phases](docs/media/pipeline.jpg)

1. **Find the plug** — the downward camera servos over the plug, then applies the calibrated camera-to-picker offset so the picker lands exactly on the grab point.
2. **Pick up** — the toolhead descends and the pneumatic vacuum gripper grabs the plug.
3. **Square the plug** — the plug is pressed against a straight edge to zero out its rotation.
4. **Inspect from below** — the machine holds the plug over the upward microscope, verifies the squaring worked (angle check), and measures precisely where the plug sits on the picker. This shift is carried into the final math, so a slightly off-center grab still ends in a perfect insertion.
5. **Find the socket** — the downward camera servos over the board socket on the phone.
6. **Insert** — the true target is computed as `socket position + nominal offset − measured grab shift`, and a slow stepped descent seats the plug into the socket.

## Hardware

<!-- IMAGE: close-up of the custom toolhead attachment + picker -->
![Toolhead and picker](docs/media/toolhead.jpg)

<!-- IMAGE: phone case holder with iPhone board in place -->
![Phone holder](docs/media/phone_holder.jpg)

- Creality Ender 3 3D printer (motion platform, stock firmware, G-code over serial)
- [Pneumatic vacuum gripper for electronics (McMaster-Carr 5083N126)](https://www.mcmaster.com/5083N126/) to pick up the plug
- Custom 3D-printed toolhead camera/gripper mount
- Custom 3D-printed camera base mount
- Custom 3D-printed iPhone holder to fixture the phone
- Downward-facing toolhead camera (1600×1200)
- Upward-facing USB microscope
- Straight-edge fixture for squaring the plug

All custom parts were designed in CAD — STEP files are in [`cad/`](cad/), ready to print.

## Machine Learning

The three models were trained on images captured by the machine itself: a grid-sweep script (`training/Picture_Grid.py`) moves the camera over the workspace and photographs the target at known coordinates, giving hundreds of labeled-position images per session. Labels were drawn as oriented rectangles/parallelograms, propagated across the sweep, and converted from CSV to YOLO OBB format (`scripts/csvToYolo.py`, `scripts/yolo_to_obb.py`).

<!-- IMAGE: example detection — camera frame with OBB drawn on the plug or socket -->
![YOLO detection example](docs/media/detection.jpg)

Oriented bounding boxes (rather than axis-aligned ones) matter here: the plug's rotation is part of the measurement, and the OBB angle feeds directly into the squaring verification.

## Repository Structure

```
├── src/                  # Final pipeline
│   ├── alignment.py      #   Full pick-and-place workflow (run this)
│   ├── Calibrate.py      #   Offset calibration helpers
│   ├── CenterPin.py, CenterSocket.py, goTo.py, debug tools
│   └── models/           #   Trained YOLOv8 weights (not in git — see Models below)
├── cad/                  # STEP files for the 3D-printed mounts and iPhone holder
├── calibration/          # Camera px→mm, pin, and table calibration scripts
├── training/             # Dataset capture (grid sweeps), labeling, earlier training iterations
├── scripts/              # Shared utilities (CSV→YOLO conversion, camera capture, jogging)
├── experiments/          # Earlier standalone experiments (pin centering, socket finding, ...)
└── docs/media/           # Photos and demo media
```

## Running It

Requires Python 3, an Ender 3 on a USB serial port, and both cameras connected.

```bash
pip install ultralytics opencv-python py3DCal

cd src
python alignment.py
```

Before a first run on new hardware, the camera-to-picker offsets and px→mm ratios in the configuration block at the top of `alignment.py` need to be recalibrated using the scripts in `calibration/`.

### Models

The trained weights (`plug_front.pt`, `plug_back.pt`, `SocketML.pt`) are excluded from git due to size. <!-- TODO: add a download link (GitHub Release is the easiest option) -->

## Results

<!-- IMAGE: the money shot — plug seated in the socket, close-up -->
![Successful insertion](docs/media/inserted.jpg)

The machine repeatably locates the plug, measures its grab error to sub-millimeter precision on the upward microscope, and seats it into the board socket. See the [demo video](YOUTUBE_LINK_HERE) for a full end-to-end run.
