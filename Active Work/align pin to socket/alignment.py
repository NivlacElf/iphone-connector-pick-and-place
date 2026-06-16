import time
import cv2
import re
import py3DCal as p3d
import sys
from ultralytics import YOLO


# --- 1. CONFIGURATION ---
PIN_MODEL_PATH = 'PinML.pt'       
SOCKET_MODEL_PATH = 'SocketML.pt' 

PRINTER_PORT = "/dev/tty.usbserial-1130"

# Camera Device Ports
CAMERA_PIN_PORT = 1    # The upward-looking microscope
CAMERA_SOCKET_PORT = 0 # The downward-looking toolhead camera

# Image Center & Hardware Offsets
CENTER_X, CENTER_Y = 800, 600
SAFE_Z_HEIGHT = 200.0
NOMINAL_OFFSET_X = 77.8
NOMINAL_OFFSET_Y = -.9
BASELINE_PIN_X = 200.680
BASELINE_PIN_Y = 198.190

# --- 2. CAMERA CONVERSION MATH ---
# Because each camera has a different focal plane and pixel-to-mm ratio, 
# we define their math separately and pass them into the alignment loop.

def pin_px_to_mm(z):
    """Returns (x_ratio, y_ratio) for the Upward Pin Camera."""
    x_ratio = (z - 125.0) / 8150.556
    y_ratio = (z - 125.0) / 8150.556
    return x_ratio, y_ratio

def pin_pixels_per_mm(z):
    """Used for drawing the 300-micron offset box."""
    # Assuming 91.58 is your offset to ground that moved over from the socket script
    return 8150.556 / (z - 125.0)

def socket_px_to_mm(z):
    """Returns (x_ratio, y_ratio) for the Downward Socket Camera."""
    x_ratio = -(z - 15.74) / 8150.556
    y_ratio = (z - 15.74) / 8150.556
    return x_ratio, y_ratio

def socket_pixels_per_mm(z):
    """Used for drawing the 300-micron offset box."""
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

# --- 4. LIVE VISION & ALIGNMENT LOOP ---
def align_and_verify(ender3, cap, model, target_name, px_to_mm_func, px_per_mm_func, invert_motion=False):
    """
    Runs the 3x alignment live while displaying bounding boxes.
    Once aligned 3 times, prompts the user to verify visually before continuing.
    Starts in a paused state until the user presses 's'.
    """
    aligned_count = 0
    iterations = 3
    is_aligning = False  # <-- Keeps the script paused initially
    final_x, final_y, final_z = get_current_pos(ender3)
    
    cv2.namedWindow(f"Live Alignment: {target_name}", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        annotated_frame = frame.copy()
        
        # --- FPS FIX: Inherit known coordinates instead of pinging the printer every frame ---
        hw_x, hw_y, hw_z = final_x, final_y, final_z

        # If AI detects a target
        if len(results[0].boxes) > 0 and hw_z is not None:
            box = results[0].boxes[0].xyxy[0] 
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            
            bw = float(x_max - x_min)
            bh = float(y_max - y_min)

            # --- DRAW BOXES ---
            # 1. Tracker Box (Green) & Confidence
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
            conf = float(results[0].boxes[0].conf[0])
            cv2.putText(annotated_frame, f"{conf * 100:.0f}%", (x_min, y_min - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 2. Exact Center Box (Blue, matches Tracker dimensions)
            c_start_x = int(CENTER_X - (bw / 2))
            c_start_y = int(CENTER_Y - (bh / 2))
            c_end_x = int(CENTER_X + (bw / 2))
            c_end_y = int(CENTER_Y + (bh / 2))
            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            # 3. 300-Micron Offset Box (Red)
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
                cv2.waitKey(1) # Force frame update before hardware freeze

                target_px_x = float((box[0] + box[2]) / 2)
                target_px_y = float((box[1] + box[3]) / 2)
                
                # Math Conversion
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
                time.sleep(1) # Settle time

                # --- FPS FIX: Update coordinates ONLY after we physically move ---
                new_pos = get_current_pos(ender3)
                if new_pos[0] is not None:
                    final_x, final_y, final_z = new_pos

                # Flush buffer so the next iteration gets the post-movement frame
                for _ in range(5): cap.read()
            
            elif not is_aligning:
                # --- Show paused status ---
                cv2.putText(annotated_frame, "PAUSED: Press 's' to begin moving", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                
            else:
                # Aligned 3x! Prompt user to confirm.
                cv2.putText(annotated_frame, "ALIGNED! Looks good?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Press 'y' to continue, 'r' to realign.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            cv2.putText(annotated_frame, "Searching for target...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw permanent coords
        cv2.putText(annotated_frame, f"X: {final_x:.2f} Y: {final_y:.2f}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Live Alignment: {target_name}", annotated_frame)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') and aligned_count >= iterations:
            print(f"User confirmed alignment for {target_name}.")
            break
        elif key == ord('s'):
            # --- Trigger to start moving ---
            print(f"\n[START] Beginning automated alignment for {target_name}...")
            is_aligning = True
        elif key == ord('r'):
            print("\n[RESET] Re-running alignment...")
            aligned_count = 0
            is_aligning = False  # --- Set back to paused upon reset
        elif key == ord('q'):
            print("\n[QUIT] Aborting script.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()
    return get_current_pos(ender3)

# --- 5. MAIN WORKFLOW ---
def main():
    print("Loading AI models...")
    pin_model = YOLO(PIN_MODEL_PATH)
    socket_model = YOLO(SOCKET_MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    # ---------------------------------------------------------
    # STEP 1: Center the Pin (Upward-Looking Microscope)
    # ---------------------------------------------------------
    print("\n--- STEP 1: LOCATING PIN ---")
    safe_move(ender3, target_x=205.0, target_y=199.0, target_z=200.0)
    
    print("Opening Upward Pin Camera...")
    cap_pin = init_camera(CAMERA_PIN_PORT)
    if cap_pin:
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
    # STEP 2: Center the Socket (Downward-Looking Toolhead Cam)
    # ---------------------------------------------------------
    print("\n--- STEP 2: LOCATING SOCKET ---")
    safe_move(ender3, target_x=15.0, target_y=103, target_z=108.0)
    
    print("Opening Downward Socket Camera...")
    cap_socket = init_camera(CAMERA_SOCKET_PORT)
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
    # STEP 3: CALCULATING OFFSETS & MATING
    # ---------------------------------------------------------
    print("\n--- STEP 3: CALCULATING OFFSETS & MATING ---")
    
    # 1. Calculate how much the attachment shifted physically
    # (Baseline minus current pin location)
    shift_x = BASELINE_PIN_X - pin_x
    shift_y = BASELINE_PIN_Y - pin_y

    print(f"Detected Attachment Shift -> X:{shift_x:+.3f}mm, Y:{shift_y:+.3f}mm")

    # 2. Add that shift to your known target calculations
    true_target_x = sock_x + NOMINAL_OFFSET_X - shift_x
    true_target_y = sock_y + NOMINAL_OFFSET_Y - shift_y

    print(f"Calculated True Target -> X:{true_target_x:.3f}, Y:{true_target_y:.3f}")
    ender3.go_to(x=true_target_x, y=true_target_y, z=sock_z)  # Final descent to mate pin/socket
    print("\n[SUCCESS] Operation Complete. Pin aligned to Socket.")
    #print("Code is paused. Press 'c' to continue, or 'q' to quit...")
    ender3.go_to(z=78)
    time.sleep(11)
    while True:
        # Ask for input, strip empty spaces, and make it lowercase
        user_key = input("Press 'c' + Enter to continue, or 'q' + Enter to quit: ").strip().lower()
        
        if user_key == 'c':
            print("Continuing...")
            break
        elif user_key == 'q':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid key. Please try again.")
    ender3.go_to(z=75)
    time.sleep(1)
    ender3.go_to(z=74)
    time.sleep(2)
    ender3.go_to(z=73)
    time.sleep(2)
    ender3.go_to(z=80)

    while True:
        # Ask for input, strip empty spaces, and make it lowercase
        user_key = input("Press 'c' + Enter to continue, or 'q' + Enter to quit: ").strip().lower()
        
        if user_key == 'c':
            print("Continuing...")
            break
        elif user_key == 'q':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid key. Please try again.")

    print("Rest of the code runs here!")
    ender3.go_to(x = sock_x + 76.9 +1.3 , y=sock_y + 1.2)
    ender3.go_to(z=74)
    ender3.go_to(z=87)
if __name__ == "__main__":
    main()