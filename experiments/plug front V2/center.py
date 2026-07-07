import time
import cv2
import re
import py3DCal as p3d
import sys
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = 'plug_front.pt'
PRINTER_PORT = "/dev/tty.usbserial-1130"
CAMERA_PORT = 0 

# Center of your image (for a 1600x1200 capture)
CENTER_X, CENTER_Y = 800, 600

# Starting coordinates for the search
START_X, START_Y, Z_HEIGHT = 5, 99.0, 108

# The physical distance between the camera lens and the picker mechanism
NOMINAL_OFFSET_X = 77.1
NOMINAL_OFFSET_Y = -0.2

# --- 2. HARDWARE FUNCTIONS ---
def get_current_pos(ender3):
    """Queries the printer hardware for its actual current position."""
    ender3.send_gcode("M114")
    # Loop to clear "ok" messages and find the coordinate string
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

# --- 3. LIVE VISION & ALIGNMENT LOOP ---
def align_and_verify(ender3, cap, model):
    """
    Runs the 3x alignment live while displaying bounding boxes.
    Starts in a paused state until the user presses 's'.
    """
    aligned_count = 0
    iterations = 3
    is_aligning = False
    
    final_x, final_y, final_z = get_current_pos(ender3)
    cv2.namedWindow("Live AI Alignment", cv2.WINDOW_NORMAL)

    print("\n[READY] Live Feed Active. Press 's' to start alignment.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        annotated_frame = frame.copy()
        
        # Inherit known coordinates instead of pinging the printer every frame for FPS
        hw_x, hw_y, hw_z = final_x, final_y, final_z

        # Only run drawing and alignment math if the AI actually sees a target
        if results[0].obb is not None and len(results[0].obb) > 0 and hw_z is not None:
            obb = results[0].obb[0]

            # xywhr: [cx, cy, w, h, angle_rad]
            xywhr = obb.xywhr[0]
            cx, cy = float(xywhr[0]), float(xywhr[1])
            bw, bh = float(xywhr[2]), float(xywhr[3])
            angle_rad = float(xywhr[4])
            angle_deg = (angle_rad * 180.0 / 3.14159265) - 90

            # Corner points for the oriented box (shape: [1, 4, 2])
            corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)  # (4, 2)

            conf = float(obb.conf[0])

            # BOX 1: THE TRACKER — rotated green box + label
            cv2.polylines(annotated_frame, [corners.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            label_x, label_y = int(corners[:, 0].min()), int(corners[:, 1].min()) - 8
            cv2.putText(annotated_frame, f"{conf * 100:.0f}%  {angle_deg:.1f}deg",
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # BOX 2: STATIC EXACT CENTER (Blue) — swap bw/bh so box is vertical
            c_start_x = int(CENTER_X - (bh / 2))
            c_start_y = int(CENTER_Y - (bw / 2))
            c_end_x = int(CENTER_X + (bh / 2))
            c_end_y = int(CENTER_Y + (bw / 2))
            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            # BOX 3: 300-MICRON OFFSET (Red)
            current_pixels_per_mm = 8150.556 / (hw_z - 91.58)
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
                cv2.imshow("Live AI Alignment", annotated_frame)
                cv2.waitKey(1) # Force frame update before hardware freeze

                # -------------------------------------------------------------
                # 1. PING BEFORE MOVE (Catches any manual adjustments you made)
                # -------------------------------------------------------------
                print("\n[SYNC] Checking hardware position BEFORE move...")
                pre_x, pre_y, pre_z = get_current_pos(ender3)
                if pre_x is not None:
                    hw_x, hw_y, hw_z = pre_x, pre_y, pre_z
                    final_x, final_y, final_z = hw_x, hw_y, hw_z # Update UI coordinates instantly

                target_px_x = cx
                target_px_y = cy
                
                # Math Conversion
                offset_px_x = target_px_x - CENTER_X
                offset_px_y = target_px_y - CENTER_Y
                
                px_to_mm_x = -(hw_z - 15.74) / 8150.556
                px_to_mm_y = (hw_z - 15.74) / 8150.556
                
                move_x = offset_px_x * px_to_mm_x
                move_y = offset_px_y * px_to_mm_y

                print(f"[ALIGNMENT] Delta -> X:{move_x:+.3f}mm, Y:{move_y:+.3f}mm")
                ender3.go_to(x=hw_x - move_x, y=hw_y - move_y)
                
                aligned_count += 1
                time.sleep(1) # Settle time

                # -------------------------------------------------------------
                # 2. PING AFTER MOVE (Confirms landing spot for the next loop)
                # -------------------------------------------------------------
                print("[SYNC] Checking hardware position AFTER move...")
                post_x, post_y, post_z = get_current_pos(ender3)
                if post_x is not None:
                    final_x, final_y, final_z = post_x, post_y, post_z

                # Flush buffer so next iteration gets post-movement frame
                for _ in range(5): cap.read()

            elif not is_aligning:
                cv2.putText(annotated_frame, "PAUSED: Press 's' to align", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                cv2.putText(annotated_frame, "ALIGNED! Looks good?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Press 'y' to lock, 'r' to realign.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "Searching for target...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw permanent coords
        cv2.putText(annotated_frame, f"X: {final_x:.2f} Y: {final_y:.2f}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Live AI Alignment", annotated_frame)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') and aligned_count >= iterations:
            print("\nUser confirmed visual alignment.")
            break
        elif key == ord('s'):
            print("\n[START] Beginning automated alignment...")
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

    return get_current_pos(ender3)

# --- 4. MAIN WORKFLOW ---
def main():
    print("Loading AI model...")
    model = YOLO(MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    print(f"Moving to initial search position...")
    ender3.go_to(x=START_X, y=START_Y, z=Z_HEIGHT)
    
    print("Initializing Camera...")
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Run the alignment protocol
    pin_x, pin_y, pin_z = align_and_verify(ender3, cap, model)
    
    # Close camera feed once aligned
    cap.release()
    cv2.destroyAllWindows()

    if pin_x is not None:
        print(f"\n=> PIN CENTERED AT: X:{pin_x:.3f}, Y:{pin_y:.3f}")
        
        # --- OFFSET FOR PICKER ---
        true_target_x = pin_x + NOMINAL_OFFSET_X
        true_target_y = pin_y + NOMINAL_OFFSET_Y

        print(f"Moving Picker Mechanism to: X:{true_target_x:.3f}, Y:{true_target_y:.3f}")
        ender3.go_to(x=true_target_x, y=true_target_y)

        print("\n[SUCCESS] Picker mechanism is hovering above the pin.")
        print("Ready for manual lowering.")
    else:
        print("[ERROR] Lost coordinate sync with printer during alignment.")

if __name__ == "__main__":
    main()