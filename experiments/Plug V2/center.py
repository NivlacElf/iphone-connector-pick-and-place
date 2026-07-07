import time
import cv2
import re
import py3DCal as p3d
import sys
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = 'plug_back.pt'
PRINTER_PORT = "/dev/tty.usbserial-1130"
CAMERA_PORT = 1

CENTER_X, CENTER_Y = 800, 600
SAFE_Z_HEIGHT = 200.0

START_X, START_Y, START_Z = 205.0, 199.0, 200.0

# --- 2. CAMERA CONVERSION MATH ---
def px_to_mm(z):
    """Returns (x_ratio, y_ratio) for the upward pin camera."""
    ratio = (z - 125.0) / 8150.556
    return ratio, ratio

def pixels_per_mm(z):
    """Used for drawing the 300-micron offset box."""
    return 8150.556 / (z - 125.0)

# --- 3. HARDWARE FUNCTIONS ---
def get_current_pos(ender3):
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

def init_camera(port_num):
    cap = cv2.VideoCapture(port_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera on port {port_num}.")
        return None
    return cap

# --- 4. LIVE VISION & ALIGNMENT LOOP (OBB) ---
def align_and_verify(ender3, cap, model):
    """
    3x iterative alignment using OBB detections.
    Paused until user presses 's'. Confirms with 'y', resets with 'r', quits with 'q'.
    Motion is inverted because the upward camera moves opposite to the toolhead.
    """
    aligned_count = 0
    iterations = 3
    is_aligning = False
    final_x, final_y, final_z = get_current_pos(ender3)

    cv2.namedWindow("Live Alignment: PIN", cv2.WINDOW_NORMAL)
    print("\n[READY] Live Feed Active. Press 's' to start alignment.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.7, verbose=False)
        annotated_frame = frame.copy()
        hw_x, hw_y, hw_z = final_x, final_y, final_z

        if results[0].obb is not None and len(results[0].obb) > 0 and hw_z is not None:
            first_obb = results[0].obb[0]

            # xywhr: [cx, cy, w, h, angle_rad]
            xywhr = first_obb.xywhr[0]
            cx, cy = float(xywhr[0]), float(xywhr[1])
            bw, bh = float(xywhr[2]), float(xywhr[3])
            angle_deg = float(xywhr[4]) * 180.0 / 3.14159265 - 90.0

            corners = first_obb.xyxyxyxy[0].cpu().numpy().astype(int)  # (4, 2)

            conf = float(first_obb.conf[0])

            # BOX 1: Rotated OBB tracker (Green)
            cv2.polylines(annotated_frame, [corners.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            label_x = int(corners[:, 0].min())
            label_y = int(corners[:, 1].min()) - 8
            cv2.putText(annotated_frame, f"{conf * 100:.0f}%  {angle_deg:.1f}deg",
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # BOX 2: Static center reference — vertical (swap bw/bh on axes)
            c_start_x = int(CENTER_X - (bh / 2))
            c_start_y = int(CENTER_Y - (bw / 2))
            c_end_x   = int(CENTER_X + (bh / 2))
            c_end_y   = int(CENTER_Y + (bw / 2))
            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            # BOX 3: 300-micron tolerance ring (Red)
            offset_px = int(0.3 * pixels_per_mm(hw_z))
            cv2.rectangle(annotated_frame,
                          (c_start_x - offset_px, c_start_y - offset_px),
                          (c_end_x   + offset_px, c_end_y   + offset_px),
                          (0, 0, 255), 2)

            # --- HARDWARE MOVEMENT ---
            if is_aligning and aligned_count < iterations:
                cv2.putText(annotated_frame, f"Aligning... ({aligned_count}/{iterations})",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("Live Alignment: PIN", annotated_frame)
                cv2.waitKey(1)

                print("\n[SYNC] Checking hardware position BEFORE move...")
                pre_x, pre_y, pre_z = get_current_pos(ender3)
                if pre_x is not None:
                    hw_x, hw_y, hw_z = pre_x, pre_y, pre_z
                    final_x, final_y, final_z = hw_x, hw_y, hw_z

                offset_px_x = cx - CENTER_X
                offset_px_y = cy - CENTER_Y
                px_x_ratio, px_y_ratio = px_to_mm(hw_z)

                # Invert motion: upward camera is mirrored relative to toolhead movement
                move_x = -(offset_px_x * px_x_ratio)
                move_y = -(offset_px_y * px_y_ratio)

                print(f"[PIN ALIGN] Delta -> X:{move_x:+.3f}mm, Y:{move_y:+.3f}mm")
                ender3.go_to(x=hw_x - move_x, y=hw_y - move_y)

                aligned_count += 1
                time.sleep(1)

                print("[SYNC] Checking hardware position AFTER move...")
                post_x, post_y, post_z = get_current_pos(ender3)
                if post_x is not None:
                    final_x, final_y, final_z = post_x, post_y, post_z

                for _ in range(5): cap.read()

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

        cv2.putText(annotated_frame, f"X: {final_x:.2f} Y: {final_y:.2f}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Alignment: PIN", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') and aligned_count >= iterations:
            print("\nUser confirmed pin alignment.")
            break
        elif key == ord('s'):
            print("\n[START] Beginning automated alignment...")
            is_aligning = True
        elif key == ord('r'):
            print("\n[RESET] Re-running alignment...")
            aligned_count = 0
            is_aligning = False
        elif key == ord('q'):
            print("\n[QUIT] Aborting.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    cv2.destroyAllWindows()
    return get_current_pos(ender3)

# --- 5. MAIN WORKFLOW ---
def main():
    print("Loading AI model...")
    model = YOLO(MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    print("Moving to pin centering position...")
    ender3.go_to(x=START_X, y=START_Y, z=START_Z)

    print("Opening upward pin camera...")
    cap = init_camera(CAMERA_PORT)
    if not cap:
        sys.exit()

    pin_x, pin_y, pin_z = align_and_verify(ender3, cap, model)
    cap.release()

    if pin_x is not None:
        print(f"\n=> PIN CENTERED AT: X:{pin_x:.3f}, Y:{pin_y:.3f}, Z:{pin_z:.3f}")
    else:
        print("[ERROR] Lost coordinate sync with printer during alignment.")

if __name__ == "__main__":
    main()
