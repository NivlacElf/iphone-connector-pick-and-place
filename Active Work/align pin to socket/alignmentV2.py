import time
import cv2
import re
import py3DCal as p3d
import sys
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
PICKUP_MODEL_PATH = 'plug_front.pt'   # Used in Phase 1
PIN_MODEL_PATH = 'plug_back.pt'       # Used in Phase 3
SOCKET_MODEL_PATH = 'SocketML.pt' # Used in Phase 4

PRINTER_PORT = "/dev/tty.usbserial-1130"

# Camera Device Ports
CAMERA_DOWNWARD_PORT = 0 # Toolhead camera (Used for initial pickup & final socket finding)
CAMERA_UPWARD_PORT = 1   # Microscope (Used to check pin alignment on picker)

# Image Center & Hardware Offsets
CENTER_X, CENTER_Y = 800, 600
SAFE_Z_HEIGHT = 200.0
NOMINAL_OFFSET_X = 77.8
NOMINAL_OFFSET_Y = -0.9

# Phase 1: Initial Pin Search coords
START_X, START_Y, START_Z = 13.0, 103.0, 108.0

# Phase 3: Baseline coordinates for the upward pin check
BASELINE_PIN_X = 200.680
BASELINE_PIN_Y = 198.190

# Phase 2.5: Squaring the plug against a straight edge
PICKUP_Z = 76.0              # Z height at which the picker grabs the plug
LIFT_AFTER_PICKUP = 10.0     # mm to retract straight up after grabbing / before traversing
PICKUP_LEFT_FRACTION = 0.25  # grab this fraction of the plug width LEFT of center ("center-left")

SQUARE_START = (57.0, 119.0, 72.0)  # staging pose: up, over, then down to here
SQUARE_APPROACH_X = 63.0            # first slide toward the straight edge
SQUARE_PUSH_END_X = 68.4            # creep into the edge until here
SQUARE_STEP = 0.3                  # mm per creep step
SQUARE_STEP_PAUSE = 0.3            # seconds to dwell between creep steps
SQUARE_RETRACT_X = 64.0            # back off to here after pressing (absolute)

# Angle verification (upward OBB microscope)
OBB_ANGLE_OFFSET = 90.0    # model reports ~90 deg when the pin is physically at 0 deg
ANGLE_TOLERANCE = 10.0     # max allowed |angle| in degrees on either side
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

def safe_move(ender3, target_x, target_y, target_z, safe_z=SAFE_Z_HEIGHT):
    """Moves Z up first, then X/Y, then Z down to avoid hitting apparatus."""
    print(f"\n[SAFE MOVE] 1. Retracting Z to safe height: {safe_z}")
    ender3.go_to(z=safe_z)
    time.sleep(1) 
    
    print(f"[SAFE MOVE] 2. Traversing to X:{target_x}, Y:{target_y}")
    ender3.go_to(x=target_x, y=target_y)
    time.sleep(2) 
    
    print(f"[SAFE MOVE] 3. Descending to Z:{target_z}")
    ender3.go_to(z=target_z)
    time.sleep(1) 

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

# --- 4. LIVE VISION & ALIGNMENT LOOP ---
def align_and_verify(ender3, cap, model, target_name, px_to_mm_func, px_per_mm_func, invert_motion=False):
    """
    Runs the 3x alignment live while displaying bounding boxes.
    Includes hardware pings before and after moves to catch manual adjustments.
    """
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

            # Draw Boxes
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
            
            l_start_x = c_start_x - offset_px
            l_start_y = c_start_y - offset_px
            l_end_x = c_end_x + offset_px
            l_end_y = c_end_y + offset_px
            cv2.rectangle(annotated_frame, (l_start_x, l_start_y), (l_end_x, l_end_y), (0, 0, 255), 2)

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
                
                aligned_count += 1
                time.sleep(1) 

                # 2. PING AFTER MOVE
                print(f"[SYNC] Checking hardware position AFTER move...")
                post_x, post_y, post_z = get_current_pos(ender3)
                if post_x is not None:
                    final_x, final_y, final_z = post_x, post_y, post_z

                # Flush buffer
                for _ in range(5): cap.read()
            
            elif not is_aligning:
                cv2.putText(annotated_frame, "PAUSED: Press 's' to begin moving", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                cv2.putText(annotated_frame, "ALIGNED! Looks good?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Press 'y' to continue, 'r' to realign.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            cv2.putText(annotated_frame, "Searching for target...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(annotated_frame, f"X: {final_x:.2f} Y: {final_y:.2f}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)

        # Keyboard Controls
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
    time.sleep(1)

    # 2. Traverse over to the staging X/Y, then descend to the staging Z (57, 119, 72)
    ender3.go_to(x=SQUARE_START[0], y=SQUARE_START[1])
    time.sleep(2)
    ender3.go_to(z=SQUARE_START[2])
    time.sleep(1)

    # 3. Slide toward the straight edge
    print(f"[SQUARE] Approaching edge to X:{SQUARE_APPROACH_X}")
    ender3.go_to(x=SQUARE_APPROACH_X)
    time.sleep(1)

    # 4. Creep into the edge in 0.3mm steps, dwelling between each
    x = SQUARE_APPROACH_X
    while x < SQUARE_PUSH_END_X - 1e-9:
        x = round(min(x + SQUARE_STEP, SQUARE_PUSH_END_X), 3)
        ender3.go_to(x=x)
        time.sleep(SQUARE_STEP_PAUSE)
    print(f"[SQUARE] Pressed into edge, reached X:{x:.3f}")

    # 5. Back off to the left, then lift straight up before traversing anywhere
    ender3.go_to(x=SQUARE_RETRACT_X)
    time.sleep(1)
    ender3.go_to(z=SQUARE_START[2] + LIFT_AFTER_PICKUP)
    time.sleep(1)
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
        pickup_x, pickup_y, pickup_z = align_and_verify(
            ender3=ender3,
            cap=cap_down,
            model=pickup_model,
            target_name="INITIAL PIN",
            px_to_mm_func=socket_px_to_mm,
            px_per_mm_func=socket_pixels_per_mm,
            invert_motion=False
        )

        # Measure the plug width (in mm) from the centered detection so we can grab
        # it center-LEFT -- we will be pushing its right side into the straight edge.
        left_offset_mm = 0.0
        if pickup_z is not None:
            for _ in range(5):
                cap_down.read()  # flush stale frames
            ret, frame = cap_down.read()
            if ret:
                box, _ = get_primary_detection(pickup_model(frame, conf=0.7, verbose=False)[0])
                if box is not None:
                    bw_px = float(box[2] - box[0])
                    x_ratio, _ = socket_px_to_mm(pickup_z)
                    width_mm = abs(bw_px * x_ratio)
                    left_offset_mm = width_mm * PICKUP_LEFT_FRACTION
                    print(f"[PICKUP] Plug width ~{width_mm:.2f}mm -> grabbing {left_offset_mm:.2f}mm left of center")

        cap_down.release()
        cv2.destroyAllWindows()

        if pickup_x is not None:
            true_target_x = pickup_x + NOMINAL_OFFSET_X +.6 - left_offset_mm
            true_target_y = pickup_y + NOMINAL_OFFSET_Y +1
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

        pin_x, pin_y, pin_z = align_and_verify(
            ender3=ender3, 
            cap=cap_pin, 
            model=pin_model, 
            target_name="PIN", 
            px_to_mm_func=pin_px_to_mm, 
            px_per_mm_func=pin_pixels_per_mm, 
            invert_motion=True
        )
        print(f"\n=> PIN FINALIZED AT: X:{pin_x:.3f}, Y:{pin_y:.3f}, Z:{pin_z:.3f}")
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
        sock_x, sock_y, sock_z = align_and_verify(
            ender3=ender3, 
            cap=cap_socket, 
            model=socket_model, 
            target_name="SOCKET", 
            px_to_mm_func=socket_px_to_mm, 
            px_per_mm_func=socket_pixels_per_mm, 
            invert_motion=False
        )
        print(f"\n=> SOCKET FINALIZED AT: X:{sock_x:.3f}, Y:{sock_y:.3f}, Z:{sock_z:.3f}")
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