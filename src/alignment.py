import time
import cv2
import re
import math
import py3DCal as p3d
import sys
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
PICKUP_MODEL_PATH = 'models/plug_front.pt'   # Used in Phase 1
PIN_MODEL_PATH = 'models/plug_back.pt'       # Used in Phase 3
SOCKET_MODEL_PATH = 'models/SocketML.pt' # Used in Phase 4

PRINTER_PORT = "/dev/tty.usbserial-1130"

# Camera Device Ports
CAMERA_DOWNWARD_PORT = 0 # Toolhead camera (Used for initial pickup & final socket finding)
CAMERA_UPWARD_PORT = 1   # Microscope (Used to check pin alignment on picker)

# Image Center & Hardware Offsets
CENTER_X, CENTER_Y = 800, 600
SAFE_Z_HEIGHT = 200.0
NOMINAL_OFFSET_X = 78.2
NOMINAL_OFFSET_Y = -0.8

# Camera-to-picker offset used specifically for the initial plug pickup.
# Kept separate from NOMINAL_OFFSET so recalibrating for socket mating doesn't affect pickup.
PICKUP_NOMINAL_OFFSET_X = 77.6
PICKUP_NOMINAL_OFFSET_Y = 1.3

# Phase 1: Initial Pin Search coords
START_X, START_Y, START_Z = 10.0, 101.0, 108.0

# Phase 3: Baseline coordinates for the upward pin check
BASELINE_PIN_X = 200.810
BASELINE_PIN_Y = 199.540

# Phase 2.5: Squaring the plug against a straight edge
PICKUP_Z = 75.0              # Z height at which the picker grabs the plug
LIFT_AFTER_PICKUP = 10.0     # mm to retract straight up after grabbing / before traversing
PICKUP_LEFT_FRACTION = .03  # grab this fraction of the plug width LEFT of center ("center-left")
PICKUP_DOWN_FRACTION = 0  # grab this fraction of the plug length BELOW center (toward bottom end)

# Display-only correction: the picker physically grabs a bit LEFT of where the marker is
# drawn (camera X), so shift JUST the on-screen GRAB crosshair to match reality. This does
# NOT change where the picker actually moves. Negative = draw further left.
GRAB_DRAW_OFFSET_X_PX = 0

# Edge-press drift compensation: when the plug is tilted, squaring it against the straight
# edge slides the grab point along the plug's length (the head pushes along the printer's X,
# not the plug's). Clockwise (negative angle) drifts the grab UP the plug, counter-clockwise
# (positive angle) drifts it DOWN. We pre-bias the grab target the opposite way, scaled
# linearly with tilt: full PICKUP_ANGLE_COMP_FRACTION of the plug LENGTH at +/-MAX_DEG.
PICKUP_ANGLE_COMP_FRACTION = 0.18      # max extra shift along the plug length (fraction of length)
PICKUP_ANGLE_LEFT_COMP_FRACTION = 0.06 # max extra shift along the plug width (fraction of width) at max angle
PICKUP_ANGLE_COMP_MAX_DEG = 45.0       # tilt magnitude (deg) that maps to the full compensation

SQUARE_START = (57.0, 119.0, 72.0)  # staging pose: up, over, then down to here
SQUARE_APPROACH_X = 62.0            # first slide toward the straight edge
SQUARE_PUSH_END_X = 65.3            # creep into the edge until here
SQUARE_STEP = 0.2                  # mm per creep step
SQUARE_STEP_PAUSE = 0.3            # seconds to dwell between creep steps
SQUARE_RETRACT_X = 62.0            # back off to here after pressing (absolute)
SQUARE_SLIDE_FEEDRATE = 60         # mm/min (= 1 mm/s) smooth, slow creep into the edge
SQUARE_TRAVERSE_FEEDRATE = 3000    # mm/min to restore normal speed after the slow slide

# Angle verification (upward OBB microscope)
OBB_ANGLE_OFFSET = 90.0    # model reports ~90 deg when the pin is physically at 0 deg
ANGLE_TOLERANCE = 5.0     # max allowed |angle| in degrees on either side
ANGLE_CLEAR_CONF = 0.90    # min confidence before trusting a reading (rejects partial views)
MIN_PIN_PX = 150           # min OBB width/height in px before trusting (rejects partial views)
ANGLE_STABLE_FRAMES = 15   # number of clear readings to collect before judging
ANGLE_STABLE_SPREAD = 5.0  # max spread (deg) across those readings to be considered settled

# --- 2. CAMERA CONVERSION MATH ---
def pin_px_to_mm(z):
    """Returns (x_ratio, y_ratio) for the Upward Pin Camera."""
    x_ratio = (z - 125.0) / 8150.556
    y_ratio = (z - 125.0) / 8150.556
    return x_ratio, y_ratio

def pin_pixels_per_mm(z):
    """Used for drawing the 300-micron offset box (Upward Cam)."""
    return 8150.556 / (z - 125.0)

def socket_px_to_mm(z):
    """Returns (x_ratio, y_ratio) for the Downward Camera (used for both Pin pickup and Socket finding)."""
    x_ratio = -(z - 15.74) / 8150.556
    y_ratio = (z - 15.74) / 8150.556
    return x_ratio, y_ratio

def socket_pixels_per_mm(z):
    """Used for drawing the 300-micron offset box (Downward Cam)."""
    return 8150.556 / (z - 15.74)

# --- 3. HARDWARE FUNCTIONS ---
def get_current_pos(ender3):
    """Queries the printer hardware for its actual current position."""
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

def wait_until_idle(ender3):
    """Block the Python script until the printer has PHYSICALLY finished every queued move.

    go_to()/send_gcode() are fire-and-forget: they only write bytes to the serial port and
    return immediately, while the move runs asynchronously on the printer. That is why a
    time.sleep() placed right after a move starts counting BEFORE the move is done.

    M400 makes Marlin wait until its motion planner is empty before continuing, and the
    M118 echo below is processed only AFTER that. So reading lines until the unique token
    comes back is a true 'wait until the toolhead has stopped'. (A bare send_gcode("M400")
    does NOT pause Python, because this library never reads the printer's reply.)"""
    ender3.send_gcode("M400")             # finish all queued moves before continuing
    ender3.send_gcode("M118 MOVE_DONE")   # echoed back only once M400 has completed
    while "MOVE_DONE" not in ender3.get_response():
        pass

def safe_move(ender3, target_x, target_y, target_z, safe_z=SAFE_Z_HEIGHT):
    """Moves Z up first, then X/Y, then Z down to avoid hitting apparatus."""
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
    """Return (xyxy_box, conf) for the first detection from either an OBB model
    (results[0].obb) or a standard detect model (results[0].boxes). (None, None) if empty.
    Using .xyxy keeps the axis-aligned box that the centering logic expects, regardless
    of which model type produced it."""
    dets = result.obb if result.obb is not None else result.boxes
    if dets is None or len(dets) == 0:
        return None, None
    return dets.xyxy[0], float(dets.conf[0])

def normalize_angle(deg):
    """Wrap an angle into (-90, 90]. Pin orientation is symmetric to 180 deg flips,
    so anything outside this band is the same physical tilt."""
    return (deg + 90.0) % 180.0 - 90.0

def get_obb_angle_deg(result):
    """Signed tilt of the part's LONG axis away from vertical (image-up), in degrees.

    Continuous through 0: a small rotation reads a small angle (1, 2, 3 ...) instead of
    jumping to +/-90 the way the raw OBB rotation does near vertical. Sign convention:
    NEGATIVE = clockwise on screen, POSITIVE = counter-clockwise. Valid for tilts up to
    +/-90 deg. Returns None if there is no OBB detection."""
    if result.obb is None or len(result.obb) == 0:
        return None
    corners = result.obb.xyxyxyxy[0].cpu().numpy().astype(float)
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    len1 = float((e1[0] ** 2 + e1[1] ** 2) ** 0.5)
    len2 = float((e2[0] ** 2 + e2[1] ** 2) ** 0.5)
    long_edge = e1 if len1 >= len2 else e2
    n = float((long_edge[0] ** 2 + long_edge[1] ** 2) ** 0.5)
    if n == 0:
        return None
    lx, ly = long_edge[0] / n, long_edge[1] / n
    if ly < 0:                      # force the long axis to point DOWN so it is single-valued
        lx, ly = -lx, -ly
    return math.degrees(math.atan2(lx, ly))

def get_obb_grab_offset_px(result, left_frac, down_frac=0.0,
                           angle_comp_frac=PICKUP_ANGLE_COMP_FRACTION,
                           angle_left_comp_frac=PICKUP_ANGLE_LEFT_COMP_FRACTION,
                           angle_comp_max_deg=PICKUP_ANGLE_COMP_MAX_DEG):
    """Pixel coords of the grab point on the plug, measured ALONG THE PLUG'S OWN AXES:
      * left_frac of the plug's WIDTH (short side) left of center, and
      * down_frac of the plug's LENGTH (long side) below center (toward the bottom end).

    The grab point is the spot that would be left-of-center / below-center if the plug
    were perfectly parallel (0 deg). Because we measure along the plug's rotated short
    and long axes (from the OBB corners) instead of straight along screen X/Y, the grab
    tracks that SAME physical spot no matter how the plug is rotated. We push the plug's
    right side into the straight edge, so we deliberately grab on the left.

    Edge-press drift compensation: clockwise tilt (negative angle) makes the squaring
    press slide the grab UP the plug, so we bias the target DOWN; counter-clockwise biases
    it UP. The bias scales linearly with tilt, reaching angle_comp_frac of the plug LENGTH
    at +/-angle_comp_max_deg.

    Returns (grab_x_px, grab_y_px, width_px) or None if there is no OBB detection."""
    if result.obb is None or len(result.obb) == 0:
        return None
    corners = result.obb.xyxyxyxy[0].cpu().numpy().astype(float)
    center = corners.mean(axis=0)

    # Two edges that share corner 0 -- shorter one is WIDTH, longer one is LENGTH.
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    len1 = float((e1[0] ** 2 + e1[1] ** 2) ** 0.5)
    len2 = float((e2[0] ** 2 + e2[1] ** 2) ** 0.5)
    if len1 <= len2:
        short_edge, width_px, long_edge, length_px = e1, len1, e2, len2
    else:
        short_edge, width_px, long_edge, length_px = e2, len2, e1, len1
    if width_px == 0 or length_px == 0:
        return None

    short_unit = short_edge / width_px
    long_unit = long_edge / length_px
    # Point the short axis toward image-left and the long axis toward image-down so we
    # always grab left-of-center / below-center when parallel (and rotate with the plug).
    if short_unit[0] > 0:
        short_unit = -short_unit
    if long_unit[1] < 0:
        long_unit = -long_unit

    # Tilt angle along the long axis (same sign convention as get_obb_angle_deg:
    # negative = clockwise). Compensate the opposite direction, clamped to +/-max.
    angle = math.degrees(math.atan2(float(long_unit[0]), float(long_unit[1])))
    clamped = max(-angle_comp_max_deg, min(angle_comp_max_deg, angle))
    comp_frac = -(clamped / angle_comp_max_deg) * angle_comp_frac
    effective_down = down_frac + comp_frac

    # At high angles the picker needs to reach further left along the plug's width axis.
    # Scale extra left offset by |angle| so it's zero when straight and maxes at full tilt.
    left_comp_frac = (abs(clamped) / angle_comp_max_deg) * angle_left_comp_frac
    effective_left = left_frac + left_comp_frac

    grab = center + short_unit * (effective_left * width_px) + long_unit * (effective_down * length_px)
    return float(grab[0]), float(grab[1]), width_px

# --- 4. LIVE VISION & ALIGNMENT LOOP ---
def align_and_verify(ender3, cap, model, target_name, px_to_mm_func, invert_motion=False,
                     grab_fraction=None, grab_down_fraction=0.0, angle_offset=0.0):
    """
    Runs the 3x alignment live while displaying bounding boxes.
    Includes hardware pings before and after moves to catch manual adjustments.

    If grab_fraction is set (only used when looking at the plug for pickup), the live
    view also draws the grab target -- the spot grab_fraction of the plug WIDTH left of
    center along the plug's own short axis -- as a red crosshair, so the user can verify
    where the picker will grab before continuing. The grab target is AVERAGED over recent
    frames and RETURNED (as the 4th value) so the actual pickup move uses the exact point
    shown on screen instead of re-detecting a different, jittery frame.

    angle_offset is subtracted from the OBB rotation before display so the on-screen angle
    reads ~0 deg when the part is parallel.

    Returns (x, y, z, grab) where grab is (grab_x_px, grab_y_px, width_px) or None.
    """
    aligned_count = 0
    iterations = 3
    is_aligning = False
    current_grab = None  # this frame's grab point (grab_x_px, grab_y_px, width_px)
    last_grab = None     # grab point locked in when the user confirms ('y'); returned to caller

    # Retry initial position — printer may still be settling after a prior step (e.g. angle check)
    final_x, final_y, final_z = None, None, None
    for _ in range(5):
        final_x, final_y, final_z = get_current_pos(ender3)
        if final_x is not None:
            break
        time.sleep(0.5)

    _pos_tick = 0  # throttle counter for mid-loop position retries
    cv2.namedWindow(f"Live Alignment: {target_name}", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        annotated_frame = frame.copy()

        hw_x, hw_y, hw_z = final_x, final_y, final_z

        # If position still unknown, retry ~once per second so we recover automatically
        if final_z is None:
            _pos_tick += 1
            if _pos_tick % 30 == 0:
                px, py, pz = get_current_pos(ender3)
                if px is not None:
                    final_x, final_y, final_z = px, py, pz
                    hw_x, hw_y, hw_z = px, py, pz

        result0 = results[0]
        is_obb = result0.obb is not None and len(result0.obb) > 0
        is_det = result0.boxes is not None and len(result0.boxes) > 0

        if is_obb or is_det:
            if is_obb:
                obb = result0.obb[0]
                xywhr = obb.xywhr[0]
                bw, bh = float(xywhr[2]), float(xywhr[3])
                conf = float(obb.conf[0])
                corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)

                # Rotated green box
                cv2.polylines(annotated_frame, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                label_x = int(corners[:, 0].min())
                label_y = int(corners[:, 1].min()) - 8
                cv2.putText(annotated_frame, f"{conf * 100:.0f}%",
                            (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Live angle readout: continuous tilt from vertical (0 deg = parallel),
                # counts 1,2,3... as it rotates instead of jumping to +/-90. Negative = CW.
                obb_angle = get_obb_angle_deg(result0)
                if obb_angle is not None:
                    obb_angle = normalize_angle(obb_angle - angle_offset)
                    cv2.putText(annotated_frame, f"angle: {obb_angle:+.1f} deg", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Live grab-target overlay (pickup phase only): red crosshair on the spot
                # the picker will grab, measured along the plug's own short/long axes.
                # We remember THIS frame's point in current_grab and lock it in the instant
                # you press 'y', so the move uses exactly what was on screen at confirm time.
                if grab_fraction is not None:
                    current_grab = get_obb_grab_offset_px(result0, grab_fraction, grab_down_fraction)
                    if current_grab is not None:
                        # Draw the marker shifted left to reflect where the picker really
                        # grabs; current_grab itself is untouched so the move is unchanged.
                        gx = int(round(current_grab[0] + GRAB_DRAW_OFFSET_X_PX))
                        gy = int(round(current_grab[1]))
                        cv2.circle(annotated_frame, (gx, gy), 12, (0, 0, 255), -1)
                        cv2.line(annotated_frame, (gx - 25, gy), (gx + 25, gy), (0, 0, 255), 2)
                        cv2.line(annotated_frame, (gx, gy - 25), (gx, gy + 25), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "GRAB", (gx + 18, gy - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Blue center box: swap bw/bh to match physical plug orientation (same as center.py)
                c_start_x = int(CENTER_X - (bh / 2))
                c_start_y = int(CENTER_Y - (bw / 2))
                c_end_x   = int(CENTER_X + (bh / 2))
                c_end_y   = int(CENTER_Y + (bw / 2))

                # Movement centroid from xyxy — always in original image pixel space
                xyxy = obb.xyxy[0]
                cx = float((xyxy[0] + xyxy[2]) / 2)
                cy = float((xyxy[1] + xyxy[3]) / 2)
            else:
                box = result0.boxes[0].xyxy[0]
                conf = float(result0.boxes[0].conf[0])
                x_min, y_min = int(box[0]), int(box[1])
                x_max, y_max = int(box[2]), int(box[3])
                bw = float(x_max - x_min)
                bh = float(y_max - y_min)
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)

                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{conf * 100:.0f}%", (x_min, y_min - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                c_start_x = int(CENTER_X - (bw / 2))
                c_start_y = int(CENTER_Y - (bh / 2))
                c_end_x   = int(CENTER_X + (bw / 2))
                c_end_y   = int(CENTER_Y + (bh / 2))

            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            if hw_z is not None:
                # Derive box scale from the same function as movement so they always match
                px_x_ratio, px_y_ratio = px_to_mm_func(hw_z)
                offset_px = int(0.3 / abs(px_x_ratio))  # 0.3 mm -> pixels
                cv2.rectangle(annotated_frame,
                              (c_start_x - offset_px, c_start_y - offset_px),
                              (c_end_x + offset_px, c_end_y + offset_px), (0, 0, 255), 2)

                # --- HARDWARE MOVEMENT ---
                if is_aligning and aligned_count < iterations:
                    cv2.putText(annotated_frame, f"Aligning... ({aligned_count}/{iterations})",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)
                    cv2.waitKey(1)

                    # 1. PING BEFORE MOVE
                    print(f"\n[SYNC] Checking hardware position BEFORE move...")
                    pre_x, pre_y, pre_z = get_current_pos(ender3)
                    if pre_x is not None:
                        hw_x, hw_y, hw_z = pre_x, pre_y, pre_z
                        final_x, final_y, final_z = hw_x, hw_y, hw_z
                        px_x_ratio, px_y_ratio = px_to_mm_func(hw_z)  # refresh ratio with updated z

                    offset_px_x = cx - CENTER_X
                    offset_px_y = cy - CENTER_Y
                    print(f"[{target_name} ALIGN] hw_z={hw_z:.3f}  ratio={px_x_ratio:.6f} mm/px  offset=({offset_px_x:+.1f}, {offset_px_y:+.1f}) px")

                    move_x = offset_px_x * px_x_ratio
                    move_y = offset_px_y * px_y_ratio

                    if invert_motion:
                        move_x, move_y = -move_x, -move_y

                    print(f"[{target_name} ALIGN] Delta -> X:{move_x:+.3f}mm, Y:{move_y:+.3f}mm")
                    ender3.go_to(x=hw_x - move_x, y=hw_y - move_y)
                    wait_until_idle(ender3)     # block Python until the toolhead has stopped

                    aligned_count += 1
                    time.sleep(0.4)             # let the image settle after motion stops

                    # 2. PING AFTER MOVE
                    print(f"[SYNC] Checking hardware position AFTER move...")
                    post_x, post_y, post_z = get_current_pos(ender3)
                    if post_x is not None:
                        final_x, final_y, final_z = post_x, post_y, post_z

                    # Drain stale frames by WALL-CLOCK time, not a fixed count. Frames
                    # captured while the toolhead was still moving show the pin at its OLD
                    # offset; acting on one makes the next iteration re-issue almost the same
                    # correction -> double travel -> overshoot. Discard ~0.7s of frames so the
                    # next detection is from the settled position.
                    _t_drain = time.time()
                    while time.time() - _t_drain < 0.7:
                        cap.read()

                elif not is_aligning:
                    cv2.putText(annotated_frame, "PAUSED: Press 's' to begin moving", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                else:
                    cv2.putText(annotated_frame, "ALIGNED! Looks good?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(annotated_frame, "Press 'y' to continue, 'r' to realign.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, "Target found. Waiting for printer position...",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        else:
            cv2.putText(annotated_frame, "Searching for target...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        x_str = f"{final_x:.2f}" if final_x is not None else "N/A"
        y_str = f"{final_y:.2f}" if final_y is not None else "N/A"
        cv2.putText(annotated_frame, f"X: {x_str} Y: {y_str}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') and aligned_count >= iterations:
            print(f"User confirmed alignment for {target_name}.")
            last_grab = current_grab   # lock in the grab point shown at this instant
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
    x, y, z = get_current_pos(ender3)
    return x, y, z, last_grab

# --- 4b. SQUARING & ANGLE VERIFICATION ---
def square_plug(ender3):
    """Press the plug against the straight edge to remove any rotation.

    Assumes the plug is grabbed center-left and is no more than ~45 deg off, so
    creeping it into a fixed edge squares it up. Path: lift straight up, traverse
    over, descend to the staging pose, then creep into the edge in small steps."""
    print("\n=== PHASE 2.5: SQUARING PLUG AGAINST STRAIGHT EDGE ===")

    # 1. Lift straight up off the pickup point (~10mm) so we clear anything on the bed
    _, _, cur_z = get_current_pos(ender3)
    if cur_z is None:
        cur_z = PICKUP_Z
    ender3.go_to(z=cur_z + LIFT_AFTER_PICKUP)
    wait_until_idle(ender3)

    # 2. Traverse over to the staging X/Y, then descend to the staging Z (57, 119, 72)
    ender3.go_to(x=SQUARE_START[0], y=SQUARE_START[1])
    wait_until_idle(ender3)
    ender3.go_to(z=SQUARE_START[2])
    wait_until_idle(ender3)   # make sure we are all the way DOWN before sliding sideways

    # 3. Slide toward the straight edge (normal speed, just positioning)
    print(f"[SQUARE] Approaching edge to X:{SQUARE_APPROACH_X}")
    ender3.go_to(x=SQUARE_APPROACH_X)
    wait_until_idle(ender3)

    # 4. Now creep into the edge as ONE smooth, continuous move at 1 mm/s.
    # G1 with an explicit feedrate (F is in mm/min) lets the planner accelerate/decelerate
    # smoothly across the whole slide -- far smoother than many tiny stop-start steps.
    print(f"[SQUARE] Sliding into edge to X:{SQUARE_PUSH_END_X} at 1 mm/s ...")
    ender3.send_gcode(f"G1 X{SQUARE_PUSH_END_X} F{SQUARE_SLIDE_FEEDRATE}")
    wait_until_idle(ender3)   # block in Python until the slow slide PHYSICALLY finishes
    print(f"[SQUARE] Pressed into edge, reached X:{SQUARE_PUSH_END_X:.3f}")
    ender3.go_to(x=SQUARE_RETRACT_X)

    # Restore a normal feedrate so the retract + later moves are not stuck at 1 mm/s.
    # F is "sticky" in Marlin: it persists until the next command changes it.
    ender3.send_gcode(f"G1 F{SQUARE_TRAVERSE_FEEDRATE}")

    # 5. Back off to the left, then lift straight up before traversing anywhere

    wait_until_idle(ender3)
    ender3.go_to(z=SQUARE_START[2] + LIFT_AFTER_PICKUP)
    wait_until_idle(ender3)
    print("[SQUARE] Plug squared and retracted.")

def check_pin_angle(cap, model):
    """Verify the squared plug is within ANGLE_TOLERANCE degrees using the upward OBB
    microscope. Returns True to continue, False to abort.

    The microscope starts seeing the pin before it is fully/clearly in view, which gives
    bogus tilt readings, so a reading only counts when confidence and box size are high.
    We then require a run of stable readings before judging."""
    print("\n=== ANGLE CHECK: VERIFYING PLUG IS SQUARE ===")
    window = "Angle Check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    readings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=ANGLE_CLEAR_CONF, verbose=False)
        result = results[0]
        annotated = frame.copy()

        angle = None
        clear = False
        if result.obb is not None and len(result.obb) > 0:
            xywhr = result.obb.xywhr[0]
            w = float(xywhr[2])
            h = float(xywhr[3])
            r = float(xywhr[4])  # rotation in radians
            conf = float(result.obb.conf[0])

            pts = result.obb.xyxyxyxy[0].cpu().numpy().astype(int)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

            angle = normalize_angle(r * 180.0 / 3.141592653589793 - OBB_ANGLE_OFFSET)

            # Only trust a reading once the pin is clearly and fully in view
            clear = conf >= ANGLE_CLEAR_CONF and w >= MIN_PIN_PX and h >= MIN_PIN_PX
            color = (0, 255, 0) if clear else (0, 165, 255)
            cv2.putText(annotated, f"angle:{angle:+.1f} conf:{conf*100:.0f}% {'CLEAR' if clear else 'partial'}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if clear:
                readings.append(angle)
                if len(readings) > ANGLE_STABLE_FRAMES:
                    readings.pop(0)
        else:
            readings.clear()
            cv2.putText(annotated, "Waiting for clear view of pin...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        measured = None
        status = f"Collecting clear readings: {len(readings)}/{ANGLE_STABLE_FRAMES}"
        if len(readings) >= ANGLE_STABLE_FRAMES:
            spread = max(readings) - min(readings)
            if spread <= ANGLE_STABLE_SPREAD:
                measured = sorted(readings)[len(readings) // 2]  # median
                if abs(measured) <= ANGLE_TOLERANCE:
                    status = f"OK: {measured:+.1f} deg (<= {ANGLE_TOLERANCE}). Continuing..."
                else:
                    status = f"OUT OF RANGE: {measured:+.1f} deg  >>> PAUSED <<<"
            else:
                status = f"Stabilizing... spread {spread:.1f} deg"
        cv2.putText(annotated, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window, annotated)
        key = cv2.waitKey(1) & 0xFF

        if measured is not None and abs(measured) <= ANGLE_TOLERANCE:
            print(f"[ANGLE] Within tolerance: {measured:+.1f} deg. Continuing.")
            cv2.destroyWindow(window)
            return True

        if measured is not None and abs(measured) > ANGLE_TOLERANCE:
            # Out of range -> pause. Let the user fix the plug and re-measure, override, or quit.
            cv2.putText(annotated, "Fix plug + 'r' recheck | 'c' continue anyway | 'q' quit",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(window, annotated)
            print(f"[ANGLE] OUT OF RANGE ({measured:+.1f} deg). Program paused.")
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k == ord('r'):
                    print("[ANGLE] Re-checking...")
                    readings.clear()
                    break
                elif k == ord('c'):
                    print("[ANGLE] User overrode angle warning. Continuing.")
                    cv2.destroyWindow(window)
                    return True
                elif k == ord('q'):
                    print("[ANGLE] User aborted at angle check.")
                    cv2.destroyWindow(window)
                    return False

        if key == ord('q'):
            cv2.destroyWindow(window)
            return False

    cv2.destroyWindow(window)
    return False

# --- 5. MAIN WORKFLOW ---
def main():
    print("Loading AI models...")
    pickup_model = YOLO(PICKUP_MODEL_PATH)
    pin_model = YOLO(PIN_MODEL_PATH)
    socket_model = YOLO(SOCKET_MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    # ---------------------------------------------------------
    # PHASE 1: Find the Pin & Apply Pickup Offset
    # ---------------------------------------------------------
    print("\n=== PHASE 1: ALIGNING TO PICK UP PIN ===")
    print(f"Moving to initial search position...")
    ender3.go_to(x=START_X, y=START_Y, z=START_Z)
    
    print("Opening Downward Camera...")
    cap_down = init_camera(CAMERA_DOWNWARD_PORT)
    
    if cap_down:
        pickup_x, pickup_y, pickup_z, grab = align_and_verify(
            ender3=ender3,
            cap=cap_down,
            model=pickup_model,
            target_name="INITIAL PIN",
            px_to_mm_func=socket_px_to_mm,
            invert_motion=False,
            grab_fraction=PICKUP_LEFT_FRACTION,
            grab_down_fraction=PICKUP_DOWN_FRACTION
        )

        # Use the EXACT grab point that was on screen the instant you confirmed centering
        # (returned from align_and_verify), so the picker goes to what you saw -- no
        # re-detecting a different, jittery frame. Convert that point's pixel offset from
        # image center into a printer move (same px->mm convention as the centering loop).
        grab_off_x_mm = 0.0
        grab_off_y_mm = 0.0
        if pickup_z is not None and grab is not None:
            grab_x_px, grab_y_px, width_px = grab
            x_ratio, y_ratio = socket_px_to_mm(pickup_z)
            grab_off_x_mm = -(grab_x_px - CENTER_X) * x_ratio
            grab_off_y_mm = -(grab_y_px - CENTER_Y) * y_ratio
            print(f"[PICKUP] Plug width ~{abs(width_px * x_ratio):.2f}mm -> grab offset "
                  f"({grab_off_x_mm:+.2f}, {grab_off_y_mm:+.2f})mm from center")

        cap_down.release()
        cv2.destroyAllWindows()

        if pickup_x is not None:
            true_target_x = pickup_x + PICKUP_NOMINAL_OFFSET_X + grab_off_x_mm
            true_target_y = pickup_y + PICKUP_NOMINAL_OFFSET_Y + grab_off_y_mm
            print(f"\nMoving Picker Mechanism to: X:{true_target_x:.3f}, Y:{true_target_y:.3f}")
            ender3.go_to(x=true_target_x, y=true_target_y)
        else:
            print("[ERROR] Lost coordinate sync. Exiting.")
            sys.exit()
    else: 
        sys.exit()

    # ---------------------------------------------------------
    # PHASE 2: Pause for Manual Pickup
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("[SUCCESS] Picker mechanism is hovering above the pin.")
    print(">>> PLEASE MANUALLY LOWER Z-AXIS TO PICK UP THE PIN <<<")
    print("="*50 + "\n")
    ender3.go_to(z=PICKUP_Z)
    time.sleep(2)

    # ---------------------------------------------------------
    # PHASE 2.5: Square the plug against the straight edge
    # ---------------------------------------------------------
    square_plug(ender3)

    # while True:
    #     user_key = input("Press 'c' + Enter to CONTINUE (Run Mating Sequence), or 'q' + Enter to QUIT: ").strip().lower()
    #     if user_key == 'c':
    #         print("\nContinuing to Phase 3...")
    #         break
    #     elif user_key == 'q':
    #         print("Exiting...")
    #         sys.exit()
    #     else:
    #         print("Invalid input. Please type 'c' or 'q'.")

    # ---------------------------------------------------------
    # PHASE 3: Center the Pin (Upward-Looking Microscope)
    # ---------------------------------------------------------
    print("\n=== PHASE 3: CENTERING PIN ON TOOLHEAD ===")
    safe_move(ender3, target_x=205.0, target_y=199.0, target_z=200.0)
    
    print("Opening Upward Pin Camera...")
    cap_pin = init_camera(CAMERA_UPWARD_PORT)
    if cap_pin:
        # Verify the squaring worked before doing anything else with the plug.
        if not check_pin_angle(cap_pin, pin_model):
            print("[ABORT] Plug angle check failed. Stopping.")
            cap_pin.release()
            cv2.destroyAllWindows()
            sys.exit()

        pin_x, pin_y, pin_z, _ = align_and_verify(
            ender3=ender3,
            cap=cap_pin,
            model=pin_model,
            target_name="PIN",
            px_to_mm_func=pin_px_to_mm,
            invert_motion=True
            # angle_offset stays 0: the continuous long-axis angle already reads ~0 deg
            # when the pin's long axis is vertical (parallel).
        )
        if pin_x is not None:
            print(f"\n=> PIN FINALIZED AT: X:{pin_x:.3f}, Y:{pin_y:.3f}, Z:{pin_z:.3f}")
        else:
            print("\n[WARNING] PIN position could not be determined.")
        cap_pin.release() 
    else: return

    # ---------------------------------------------------------
    # PHASE 4: Center the Socket (Downward-Looking Toolhead Cam)
    # ---------------------------------------------------------
    print("\n=== PHASE 4: LOCATING SOCKET ===")
    safe_move(ender3, target_x=15.0, target_y=103.0, target_z=108.0)
    
    print("Opening Downward Socket Camera...")
    cap_socket = init_camera(CAMERA_DOWNWARD_PORT)
    if cap_socket:
        sock_x, sock_y, sock_z, _ = align_and_verify(
            ender3=ender3,
            cap=cap_socket,
            model=socket_model,
            target_name="SOCKET",
            px_to_mm_func=socket_px_to_mm,
            invert_motion=False
        )
        if sock_x is not None:
            print(f"\n=> SOCKET FINALIZED AT: X:{sock_x:.3f}, Y:{sock_y:.3f}, Z:{sock_z:.3f}")
        else:
            print("\n[WARNING] SOCKET position could not be determined.")
        cap_socket.release() 
    else: return

    # ---------------------------------------------------------
    # PHASE 5: CALCULATING OFFSETS & MATING
    # ---------------------------------------------------------
    print("\n=== PHASE 5: CALCULATING OFFSETS & MATING ===")
    
    shift_x = BASELINE_PIN_X - pin_x
    shift_y = BASELINE_PIN_Y - pin_y

    print(f"Detected Attachment Shift -> X:{shift_x:+.3f}mm, Y:{shift_y:+.3f}mm")

    true_target_x = sock_x + NOMINAL_OFFSET_X - shift_x
    true_target_y = sock_y + NOMINAL_OFFSET_Y - shift_y

    print(f"Calculated True Target -> X:{true_target_x:.3f}, Y:{true_target_y:.3f}")
    ender3.go_to(x=true_target_x, y=true_target_y, z=sock_z) 
    
    print("\nExecuting final descent protocol...")
    ender3.go_to(z=76)
    time.sleep(11)
    ender3.go_to(z=75)
    time.sleep(1)
    ender3.go_to(z=74)
    time.sleep(1)
    ender3.go_to(z=80)
    
    # Secondary alignment move
    ender3.go_to(x = sock_x + 76.9 + 1.3, y=sock_y + 1.2)
    ender3.go_to(z=74)
    ender3.go_to(z=87)

    print("\n[SUCCESS] Operation Complete. Pin aligned to Socket.")

if __name__ == "__main__":
    main()