"""
Calibrate.py  --  Recalibrate the NOMINAL_OFFSET values used by alignmentV2.py

WHY THIS EXISTS
---------------
alignmentV2 mates the plug to the socket with this Phase-5 math:

    shift_x        = BASELINE_PIN_X - pin_x          # how far the attachment drifted
    shift_y        = BASELINE_PIN_Y - pin_y
    true_target_x  = sock_x + NOMINAL_OFFSET_X - shift_x
    true_target_y  = sock_y + NOMINAL_OFFSET_Y - shift_y

NOMINAL_OFFSET_X / NOMINAL_OFFSET_Y are the fixed picker-to-camera offset. Every day
the attachment goes back on slightly differently, so that fixed offset drifts and needs
to be re-measured.

WHAT THIS SCRIPT DOES (assumes the plug is ALREADY picked up & squared on the toolhead,
i.e. exactly the state alignmentV2 is in entering Phase 3):

  1. Centers the pin on the toolhead with the upward microscope  -> pin_x, pin_y
     (gives the same `shift` alignmentV2 uses). Rotation is IGNORED on purpose.
  2. Finds the socket with the downward camera                   -> sock_x, sock_y, sock_z
  3. Moves the plug to the CURRENT computed mating target, hovering just above the
     socket at HOVER_Z.
  4. You manually jog X/Y with the arrow keys (or WASD) until the pin is PERFECTLY
     over the socket, then press 'c'.
  5. It pings the printer (M114) for the corrected position and prints the new
     NOMINAL_OFFSET_X / NOMINAL_OFFSET_Y to paste back into alignmentV2.py.

The correction is simply:  new_offset = old_offset + (your_position - computed_target)
"""

import time
import cv2
import re
import sys
import numpy as np
import py3DCal as p3d
from ultralytics import YOLO

# --- 1. CONFIGURATION (keep these identical to alignmentV2.py) ---
PIN_MODEL_PATH = 'plug_back.pt'    # upward microscope (pin centering)
SOCKET_MODEL_PATH = 'SocketML.pt'  # downward camera (socket finding)

PRINTER_PORT = "/dev/tty.usbserial-1130"

CAMERA_DOWNWARD_PORT = 0  # toolhead camera (socket finding)
CAMERA_UPWARD_PORT = 1    # microscope (pin centering)

CENTER_X, CENTER_Y = 800, 600
SAFE_Z_HEIGHT = 200.0

# >>> The values being calibrated. These MUST match what is currently in alignmentV2.py <<<
NOMINAL_OFFSET_X = 78.2
NOMINAL_OFFSET_Y = -.8

# Baseline pin position used to compute the attachment shift (must match alignmentV2.py)
BASELINE_PIN_X = 200.810
BASELINE_PIN_Y = 199.540

# Hover height for the manual alignment step. alignmentV2 mates around Z=74-76,
# so Z=80 parks the pin just above the socket. Tune if needed.
HOVER_Z = 80.0

# Phase 3 / Phase 4 staging poses (copied from alignmentV2.py)
PIN_STAGE = (205.0, 199.0, 200.0)   # over the upward microscope
SOCKET_STAGE = (15.0, 103.0, 108.0) # over the socket with the downward camera

# --- Manual jog settings ---
JOG_STEPS = [0.05, 0.1, 0.5, 1.0]   # selectable with number keys 1-4
DEFAULT_STEP_INDEX = 1              # start at 0.1mm
JOG_X_SIGN = 1.0                    # flip to -1.0 if Left/Right move the wrong way
JOG_Y_SIGN = 1.0                    # flip to -1.0 if Up/Down move the wrong way
Z_JOG_STEP = 0.2                    # mm per Z nudge (r = up, f = down)

# Arrow key codes vary by OS/backend; cover mac, linux(GTK) and windows.
KEY_UP    = {63232, 65362, 2490368}
KEY_DOWN  = {63233, 65364, 2621440}
KEY_LEFT  = {63234, 65361, 2424832}
KEY_RIGHT = {63235, 65363, 2555904}


# --- 2. CAMERA CONVERSION MATH (copied from alignmentV2.py) ---
def pin_px_to_mm(z):
    x_ratio = (z - 125.0) / 8150.556
    y_ratio = (z - 125.0) / 8150.556
    return x_ratio, y_ratio

def pin_pixels_per_mm(z):
    return 8150.556 / (z - 125.0)

def socket_px_to_mm(z):
    x_ratio = -(z - 15.74) / 8150.556
    y_ratio = (z - 15.74) / 8150.556
    return x_ratio, y_ratio

def socket_pixels_per_mm(z):
    return 8150.556 / (z - 15.74)


# --- 3. HARDWARE / VISION HELPERS (copied from alignmentV2.py) ---
def wait_until_idle(ender3):
    """Block Python until the printer has physically finished every queued move.
    See alignmentV2.py for a full explanation of why bare M400 does not pause Python."""
    ender3.send_gcode("M400")
    ender3.send_gcode("M118 MOVE_DONE")
    while "MOVE_DONE" not in ender3.get_response():
        pass

def get_current_pos(ender3):
    """Query the printer for its actual current position (M114)."""
    ender3.send_gcode("M114")
    for _ in range(10):
        time.sleep(0.1)
        response = ender3.get_response()
        if not response:
            continue
        x_match = re.search(r"X:([-+]?[0-9]*\.?[0-9]+)", response)
        y_match = re.search(r"Y:([-+]?[0-9]*\.?[0-9]+)", response)
        z_match = re.search(r"Z:([-+]?[0-9]*\.?[0-9]+)", response)
        if x_match and y_match and z_match:
            return float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1))
    return None, None, None

def safe_move(ender3, target_x, target_y, target_z, safe_z=SAFE_Z_HEIGHT):
    """Z up first, then X/Y, then Z down -- avoids dragging through the apparatus."""
    print(f"\n[SAFE MOVE] 1. Retracting Z to safe height: {safe_z}")
    ender3.go_to(z=safe_z)
    wait_until_idle(ender3)
    print(f"[SAFE MOVE] 2. Traversing to X:{target_x}, Y:{target_y}")
    ender3.go_to(x=target_x, y=target_y)
    wait_until_idle(ender3)
    print(f"[SAFE MOVE] 3. Descending to Z:{target_z}")
    ender3.go_to(z=target_z)
    wait_until_idle(ender3)

def init_camera(port_num):
    cap = cv2.VideoCapture(port_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera on port {port_num}.")
        return None
    return cap

def get_primary_detection(result):
    """(xyxy_box, conf) for the first detection from an OBB or standard detect model."""
    dets = result.obb if result.obb is not None else result.boxes
    if dets is None or len(dets) == 0:
        return None, None
    return dets.xyxy[0], float(dets.conf[0])


# --- 4. LIVE VISION & ALIGNMENT LOOP (copied from alignmentV2.py) ---
def align_and_verify(ender3, cap, model, target_name, px_to_mm_func, px_per_mm_func, invert_motion=False):
    """Same 3x auto-centering loop alignmentV2 uses. Rotation is not considered here."""
    aligned_count = 0
    iterations = 3
    is_aligning = False
    final_x, final_y, final_z = get_current_pos(ender3)

    cv2.namedWindow(f"Live Alignment: {target_name}", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        annotated_frame = frame.copy()
        hw_x, hw_y, hw_z = final_x, final_y, final_z

        box, conf = get_primary_detection(results[0])
        if box is not None and hw_z is not None:
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            bw = float(x_max - x_min)
            bh = float(y_max - y_min)

            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{conf * 100:.0f}%", (x_min, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            c_start_x = int(CENTER_X - (bw / 2))
            c_start_y = int(CENTER_Y - (bh / 2))
            c_end_x = int(CENTER_X + (bw / 2))
            c_end_y = int(CENTER_Y + (bh / 2))
            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            current_pixels_per_mm = px_per_mm_func(hw_z)
            offset_px = int(0.3 * current_pixels_per_mm)
            cv2.rectangle(annotated_frame,
                          (c_start_x - offset_px, c_start_y - offset_px),
                          (c_end_x + offset_px, c_end_y + offset_px), (0, 0, 255), 2)

            if is_aligning and aligned_count < iterations:
                cv2.putText(annotated_frame, f"Aligning... ({aligned_count}/{iterations})",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)
                cv2.waitKey(1)

                print(f"\n[SYNC] Checking hardware position BEFORE move...")
                pre_x, pre_y, pre_z = get_current_pos(ender3)
                if pre_x is not None:
                    hw_x, hw_y, hw_z = pre_x, pre_y, pre_z
                    final_x, final_y, final_z = hw_x, hw_y, hw_z

                target_px_x = float((box[0] + box[2]) / 2)
                target_px_y = float((box[1] + box[3]) / 2)
                offset_px_x = target_px_x - CENTER_X
                offset_px_y = target_px_y - CENTER_Y
                px_x_ratio, px_y_ratio = px_to_mm_func(hw_z)
                move_x = offset_px_x * px_x_ratio
                move_y = offset_px_y * px_y_ratio
                if invert_motion:
                    move_x, move_y = -move_x, -move_y

                print(f"[{target_name} ALIGN] Delta -> X:{move_x:+.3f}mm, Y:{move_y:+.3f}mm")
                ender3.go_to(x=hw_x - move_x, y=hw_y - move_y)
                wait_until_idle(ender3)
                aligned_count += 1
                time.sleep(0.4)             # let the image settle after motion stops

                print(f"[SYNC] Checking hardware position AFTER move...")
                post_x, post_y, post_z = get_current_pos(ender3)
                if post_x is not None:
                    final_x, final_y, final_z = post_x, post_y, post_z
                for _ in range(5):
                    cap.read()

            elif not is_aligning:
                cv2.putText(annotated_frame, "PAUSED: Press 's' to begin moving", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                cv2.putText(annotated_frame, "ALIGNED! Looks good?", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Press 'y' to continue, 'r' to realign.", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "Searching for target...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        x_str = f"{final_x:.2f}" if final_x is not None else "N/A"
        y_str = f"{final_y:.2f}" if final_y is not None else "N/A"
        cv2.putText(annotated_frame, f"X: {x_str} Y: {y_str}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') and aligned_count >= iterations:
            print(f"User confirmed alignment for {target_name}.")
            break
        elif key == ord('s'):
            print(f"\n[START] Beginning automated alignment for {target_name}...")
            is_aligning = True
        elif key == ord('r'):
            print("\n[RESET] Re-running alignment...")
            aligned_count = 0
            is_aligning = False
        elif key == ord('q'):
            print("\n[QUIT] Aborting script.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    cv2.destroyAllWindows()
    return get_current_pos(ender3)


# --- 5. MANUAL JOG ---
def manual_jog(ender3, start_x, start_y, start_z):
    """Let the user nudge X/Y (and optionally Z) until the pin sits perfectly over the
    socket, then ping the printer for the corrected position.

    Returns (actual_x, actual_y) read from M114 after the user confirms with 'c',
    or (None, None) if they abort with 'q'.
    """
    cur_x, cur_y, cur_z = start_x, start_y, start_z
    step_index = DEFAULT_STEP_INDEX

    window = "Manual Calibration Jog"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("\n=== MANUAL JOG ===")
    print("Arrow keys / WASD : nudge X/Y     |   r / f : Z up / down")
    print("1 2 3 4           : step size     |   c : LOCK IN (perfectly aligned)   q : abort")

    canvas_h, canvas_w = 360, 760

    while True:
        step = JOG_STEPS[step_index]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        lines = [
            ("MANUAL CALIBRATION JOG", (0, 255, 255)),
            (f"X: {cur_x:.3f}   Y: {cur_y:.3f}   Z: {cur_z:.3f}", (0, 255, 0)),
            (f"Step: {step:.2f} mm   (1=0.05  2=0.1  3=0.5  4=1.0)", (255, 255, 255)),
            ("Arrows / WASD = jog X-Y", (200, 200, 200)),
            ("r / f = Z up / down", (200, 200, 200)),
            ("c = LOCK IN aligned    q = abort", (0, 165, 255)),
        ]
        y0 = 50
        for text, color in lines:
            cv2.putText(canvas, text, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y0 += 50
        cv2.imshow(window, canvas)

        key = cv2.waitKeyEx(50)
        if key == -1:
            continue

        ascii_key = key & 0xFF
        dx = dy = dz = 0.0

        if key in KEY_LEFT or ascii_key == ord('a'):
            dx = -step * JOG_X_SIGN
        elif key in KEY_RIGHT or ascii_key == ord('d'):
            dx = step * JOG_X_SIGN
        elif key in KEY_UP or ascii_key == ord('w'):
            dy = step * JOG_Y_SIGN
        elif key in KEY_DOWN or ascii_key == ord('s'):
            dy = -step * JOG_Y_SIGN
        elif ascii_key == ord('r'):
            dz = Z_JOG_STEP
        elif ascii_key == ord('f'):
            dz = -Z_JOG_STEP
        elif ascii_key in (ord('1'), ord('2'), ord('3'), ord('4')):
            step_index = ascii_key - ord('1')
            continue
        elif ascii_key == ord('c'):
            print("[JOG] Locked in. Pinging printer for corrected position...")
            ax, ay = None, None
            for attempt in range(5):
                ax, ay, _ = get_current_pos(ender3)
                if ax is not None:
                    break
                print(f"[JOG] M114 ping failed, retrying ({attempt + 1}/5)...")
                time.sleep(0.3)
            if ax is None:
                print("[JOG] Could not read printer position after 5 attempts. Try pressing 'c' again.")
                continue  # don't close the window, let them retry
            cv2.destroyWindow(window)
            return ax, ay
        elif ascii_key == ord('q'):
            print("[JOG] Aborted by user.")
            cv2.destroyWindow(window)
            return None, None
        else:
            continue

        cur_x += dx
        cur_y += dy
        cur_z += dz
        ender3.go_to(x=cur_x, y=cur_y, z=cur_z)


# --- 6. MAIN CALIBRATION WORKFLOW ---
def main():
    print("Loading AI models...")
    pin_model = YOLO(PIN_MODEL_PATH)
    socket_model = YOLO(SOCKET_MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    print("\n" + "=" * 60)
    print("CALIBRATION ASSUMES THE PLUG IS ALREADY PICKED UP & SQUARED")
    print("on the toolhead (same state alignmentV2 is in entering Phase 3).")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Center the pin (upward microscope) -> attachment shift
    # ---------------------------------------------------------
    print("\n=== STEP 1: CENTERING PIN ON TOOLHEAD (rotation ignored) ===")
    safe_move(ender3, *PIN_STAGE)

    print("Opening Upward Pin Camera...")
    cap_pin = init_camera(CAMERA_UPWARD_PORT)
    if not cap_pin:
        sys.exit()
    pin_x, pin_y, pin_z = align_and_verify(
        ender3=ender3, cap=cap_pin, model=pin_model, target_name="PIN",
        px_to_mm_func=pin_px_to_mm, px_per_mm_func=pin_pixels_per_mm, invert_motion=True)
    cap_pin.release()
    cv2.destroyAllWindows()
    if pin_x is None:
        print("[ERROR] Lost coordinate sync during pin centering. Exiting.")
        sys.exit()
    print(f"=> PIN FINALIZED AT: X:{pin_x:.3f}, Y:{pin_y:.3f}, Z:{pin_z:.3f}")

    shift_x = BASELINE_PIN_X - pin_x
    shift_y = BASELINE_PIN_Y - pin_y
    print(f"Detected Attachment Shift -> X:{shift_x:+.3f}mm, Y:{shift_y:+.3f}mm")

    # ---------------------------------------------------------
    # STEP 2: Find the socket (downward camera)
    # ---------------------------------------------------------
    print("\n=== STEP 2: LOCATING SOCKET ===")
    safe_move(ender3, *SOCKET_STAGE)

    print("Opening Downward Socket Camera...")
    cap_socket = init_camera(CAMERA_DOWNWARD_PORT)
    if not cap_socket:
        sys.exit()
    sock_x, sock_y, sock_z = align_and_verify(
        ender3=ender3, cap=cap_socket, model=socket_model, target_name="SOCKET",
        px_to_mm_func=socket_px_to_mm, px_per_mm_func=socket_pixels_per_mm, invert_motion=False)
    cap_socket.release()
    cv2.destroyAllWindows()
    if sock_x is None:
        print("[ERROR] Lost coordinate sync during socket finding. Exiting.")
        sys.exit()
    print(f"=> SOCKET FINALIZED AT: X:{sock_x:.3f}, Y:{sock_y:.3f}, Z:{sock_z:.3f}")

    # ---------------------------------------------------------
    # STEP 3: Move the plug to the CURRENT computed mating target, hovering above socket
    # ---------------------------------------------------------
    print("\n=== STEP 3: HOVERING PLUG ABOVE SOCKET (current offsets) ===")
    true_target_x = sock_x + NOMINAL_OFFSET_X - shift_x
    true_target_y = sock_y + NOMINAL_OFFSET_Y - shift_y
    print(f"Computed target with CURRENT offsets -> X:{true_target_x:.3f}, Y:{true_target_y:.3f}")
    print(f"Hovering at Z:{HOVER_Z}")
    safe_move(ender3, true_target_x, true_target_y, HOVER_Z, safe_z=100.0)

    # ---------------------------------------------------------
    # STEP 4: Manual jog to perfect alignment, then ping for corrected position
    # ---------------------------------------------------------
    adj_x, adj_y = manual_jog(ender3, true_target_x, true_target_y, HOVER_Z)
    if adj_x is None:
        print("[ABORT] Calibration aborted before locking in. No offsets produced.")
        sys.exit()

    # ---------------------------------------------------------
    # STEP 5: Compute & report the new NOMINAL_OFFSET values
    # ---------------------------------------------------------
    delta_x = adj_x - true_target_x
    delta_y = adj_y - true_target_y
    new_offset_x = NOMINAL_OFFSET_X + delta_x
    new_offset_y = NOMINAL_OFFSET_Y + delta_y

    print("\n" + "=" * 60)
    print("CALIBRATION RESULT")
    print("=" * 60)
    print(f"Old offsets : NOMINAL_OFFSET_X = {NOMINAL_OFFSET_X:.3f}   NOMINAL_OFFSET_Y = {NOMINAL_OFFSET_Y:.3f}")
    print(f"Old baseline: BASELINE_PIN_X   = {BASELINE_PIN_X:.3f}   BASELINE_PIN_Y   = {BASELINE_PIN_Y:.3f}")
    print(f"Your manual correction : X:{delta_x:+.3f}mm   Y:{delta_y:+.3f}mm")
    print("-" * 60)
    print(">>> PASTE THESE INTO alignmentV2.py AND Calibrate.py <<<")
    print(f"NOMINAL_OFFSET_X = {new_offset_x:.3f}")
    print(f"NOMINAL_OFFSET_Y = {new_offset_y:.3f}")
    print(f"BASELINE_PIN_X   = {pin_x:.3f}")
    print(f"BASELINE_PIN_Y   = {pin_y:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
