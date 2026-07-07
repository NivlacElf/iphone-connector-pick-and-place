"""
debug_center_pin_up.py  --  Isolated debugger for centering the pin on the UPWARD microscope.

PURPOSE
-------
Phase 3 of alignmentV2 overshoots when centering the pin, even though the move math
is identical to Calibrate.py. This strips everything else away so we can see exactly
what numbers drive the move and measure the real overshoot factor.

It does ONE move per keypress (not a 3x auto-loop) and prints, every move:
    - actual camera frame resolution        (catches a 1600x1200 mismatch)
    - detected OBB center (px) vs CENTER     (the pixel offset)
    - hw_z and the mm/px ratio in use
    - commanded mm move and the before/after hardware position (real travel)

CONTROLS (focus the OpenCV window)
    s : do ONE centering move
    [ / ] : decrease / increase GAIN by 0.05   (multiplies the computed move)
    i : toggle invert_motion
    p : re-ping hardware position
    q : quit

WORKFLOW TO FIND THE BUG
    1. Run it. Note the printed frame resolution. If it is NOT 1600x1200, that is
       the bug: CENTER_X/Y and the 8150.556 ratio are calibrated for 1600x1200.
    2. Press 's' once. Compare commanded mm vs actual travel (after - before).
       If they match but the pin still overshoots the image center, the px->mm
       ratio is too large -> lower GAIN with '[' until one move lands it centered.
       Whatever GAIN centers it in one shot is your ratio-correction factor.
"""

import time
import re
import sys
import cv2
import py3DCal as p3d
from ultralytics import YOLO

# --- CONFIG (kept identical to alignmentV2.py) ---
PIN_MODEL_PATH = 'plug_back.pt'
PRINTER_PORT = "/dev/tty.usbserial-1130"
CAMERA_UPWARD_PORT = 1

CENTER_X, CENTER_Y = 800, 600          # assumes a 1600x1200 frame
REQ_W, REQ_H = 1600, 1200
SAFE_Z_HEIGHT = 200.0
PIN_STAGE = (205.0, 199.0, 200.0)      # same staging pose as alignmentV2 Phase 3

# Live-tunable so we can measure the overshoot empirically
GAIN = 1.0
INVERT_MOTION = True


def pin_px_to_mm(z):
    x_ratio = (z - 125.0) / 8150.556
    y_ratio = (z - 125.0) / 8150.556
    return x_ratio, y_ratio


def get_current_pos(ender3):
    ender3.send_gcode("M114")
    for _ in range(10):
        time.sleep(0.1)
        response = ender3.get_response()
        if not response:
            continue
        x = re.search(r"X:([-+]?[0-9]*\.?[0-9]+)", response)
        y = re.search(r"Y:([-+]?[0-9]*\.?[0-9]+)", response)
        z = re.search(r"Z:([-+]?[0-9]*\.?[0-9]+)", response)
        if x and y and z:
            return float(x.group(1)), float(y.group(1)), float(z.group(1))
    return None, None, None


def safe_move(ender3, tx, ty, tz, safe_z=SAFE_Z_HEIGHT):
    print(f"\n[SAFE MOVE] Z up to {safe_z}")
    ender3.go_to(z=safe_z); time.sleep(1)
    print(f"[SAFE MOVE] traverse to X:{tx} Y:{ty}")
    ender3.go_to(x=tx, y=ty); time.sleep(2)
    print(f"[SAFE MOVE] descend to Z:{tz}")
    ender3.go_to(z=tz); time.sleep(1)


def init_camera(port):
    cap = cv2.VideoCapture(port)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
    if not cap.isOpened():
        print(f"[ERROR] camera {port} not opened")
        return None
    return cap


def get_obb_center(result0):
    """Return (cx, cy, bw, bh, conf, corners) from the first OBB detection, else None."""
    if result0.obb is None or len(result0.obb) == 0:
        return None
    obb = result0.obb[0]
    xywhr = obb.xywhr[0]
    bw, bh = float(xywhr[2]), float(xywhr[3])
    conf = float(obb.conf[0])
    corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)
    xyxy = obb.xyxy[0]
    cx = float((xyxy[0] + xyxy[2]) / 2)
    cy = float((xyxy[1] + xyxy[3]) / 2)
    return cx, cy, bw, bh, conf, corners


def main():
    global GAIN, INVERT_MOTION

    print("Loading pin model...")
    model = YOLO(PIN_MODEL_PATH)

    print("Connecting to printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    print("Staging over the upward microscope...")
    safe_move(ender3, *PIN_STAGE)

    cap = init_camera(CAMERA_UPWARD_PORT)
    if cap is None:
        sys.exit()

    hw_x, hw_y, hw_z = get_current_pos(ender3)
    resolution_reported = False

    win = "DEBUG: Center Pin (upward)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("\nControls: s=one move  [ / ]=gain -/+  i=toggle invert  p=reping  q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Report the ACTUAL resolution once -- the #1 suspect for the overshoot.
        if not resolution_reported:
            h, w = frame.shape[:2]
            print(f"[RESOLUTION] requested {REQ_W}x{REQ_H}, actual {w}x{h}  "
                  f"(CENTER assumes {REQ_W}x{REQ_H} -> {CENTER_X},{CENTER_Y})")
            if (w, h) != (REQ_W, REQ_H):
                print("[RESOLUTION] *** MISMATCH -> CENTER and the px->mm ratio are wrong. "
                      "This alone causes overshoot. ***")
            resolution_reported = True

        results = model(frame, conf=0.7, verbose=False)
        ann = frame.copy()

        # Crosshair at the assumed image center
        cv2.drawMarker(ann, (CENTER_X, CENTER_Y), (255, 255, 0), cv2.MARKER_CROSS, 40, 2)

        det = get_obb_center(results[0])
        if det is not None and hw_z is not None:
            cx, cy, bw, bh, conf, corners = det
            cv2.polylines(ann, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            cv2.drawMarker(ann, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 2)

            rx, ry = pin_px_to_mm(hw_z)
            off_px_x = cx - CENTER_X
            off_px_y = cy - CENTER_Y
            mv_x = off_px_x * rx * GAIN
            mv_y = off_px_y * ry * GAIN
            if INVERT_MOTION:
                mv_x, mv_y = -mv_x, -mv_y

            cv2.putText(ann, f"conf {conf*100:.0f}%  off=({off_px_x:+.0f},{off_px_y:+.0f})px",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ann, f"z={hw_z:.2f} ratio={rx:.6f}mm/px gain={GAIN:.2f} inv={INVERT_MOTION}",
                        (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(ann, f"would move X{mv_x:+.3f} Y{mv_y:+.3f} mm   (press s)",
                        (40, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(ann, "no OBB detection / no hw pos", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(win, ann)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('[') :
            GAIN = max(0.05, round(GAIN - 0.05, 2))
            print(f"[GAIN] {GAIN:.2f}")
        elif key == ord(']'):
            GAIN = round(GAIN + 0.05, 2)
            print(f"[GAIN] {GAIN:.2f}")
        elif key == ord('i'):
            INVERT_MOTION = not INVERT_MOTION
            print(f"[INVERT] {INVERT_MOTION}")
        elif key == ord('p'):
            hw_x, hw_y, hw_z = get_current_pos(ender3)
            print(f"[PING] X:{hw_x} Y:{hw_y} Z:{hw_z}")
        elif key == ord('s') and det is not None:
            cx, cy, bw, bh, conf, corners = det
            print("\n--- ONE MOVE ---")
            pre_x, pre_y, pre_z = get_current_pos(ender3)
            if pre_x is not None:
                hw_x, hw_y, hw_z = pre_x, pre_y, pre_z
            rx, ry = pin_px_to_mm(hw_z)
            off_px_x = cx - CENTER_X
            off_px_y = cy - CENTER_Y
            mv_x = off_px_x * rx * GAIN
            mv_y = off_px_y * ry * GAIN
            if INVERT_MOTION:
                mv_x, mv_y = -mv_x, -mv_y

            print(f"  center px      : ({cx:.1f}, {cy:.1f})  vs CENTER ({CENTER_X}, {CENTER_Y})")
            print(f"  offset px      : ({off_px_x:+.1f}, {off_px_y:+.1f})")
            print(f"  z / ratio      : z={hw_z:.3f}  ratio={rx:.6f} mm/px  gain={GAIN:.2f}")
            print(f"  commanded move : X{mv_x:+.3f}  Y{mv_y:+.3f} mm")
            print(f"  before         : X{pre_x:.3f} Y{pre_y:.3f}")

            ender3.go_to(x=hw_x - mv_x, y=hw_y - mv_y)
            time.sleep(1.2)
            for _ in range(5):
                cap.read()  # flush buffered (stale) frames so next detection is fresh

            post_x, post_y, post_z = get_current_pos(ender3)
            if post_x is not None:
                hw_x, hw_y, hw_z = post_x, post_y, post_z
                print(f"  after          : X{post_x:.3f} Y{post_y:.3f}")
                print(f"  actual travel  : X{post_x - pre_x:+.3f} Y{post_y - pre_y:+.3f} mm "
                      f"(should match commanded move)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
