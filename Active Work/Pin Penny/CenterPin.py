import time
import cv2
import re
import py3DCal as p3d
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = 'best.pt'
PRINTER_PORT = "/dev/tty.usbserial-140"
CAMERA_PORT = 0 

# Center of your image (for a 1600x1200 capture)
CENTER_X, CENTER_Y = 800, 600

# Physical alignment ratios (mm per pixel)
# 1 / 86.2 = 0.0116009...
# Starting coordinates
START_X, START_Y, Z_HEIGHT = 90.0, 117.0, 185

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
def align_target(ender3, target_px_x, target_px_y, hardware_x, hardware_y, hardware_z):
    """Calculates and executes physical movement based on hardware coordinates."""
    offset_px_x = target_px_x - CENTER_X
    offset_px_y = target_px_y - CENTER_Y

    # Convert pixel offset to physical move
    px_to_mm_x = -(hardware_z - 15.74) / 8150.556
    px_to_mm_y = (hardware_z - 15.74) / 8150.556
    move_x = offset_px_x * px_to_mm_x
    move_y = offset_px_y * px_to_mm_y

    # Calculate the NEW absolute position using hardware baseline
    new_x = hardware_x - move_x
    new_y = hardware_y - move_y

    print(f"\n[ALIGNMENT] Hardware Pos: X:{hardware_x:.3f}, Y:{hardware_y:.3f}, Z:{hardware_z:.3f}")
    print(f"[ALIGNMENT] Move delta: X:{move_x:+.3f}mm, Y:{move_y:+.3f}mm")
    print(f"[ALIGNMENT] Moving to: X:{new_x:.3f}, Y:{new_y:.3f}")
    
    ender3.go_to(x=new_x, y=new_y)
    return new_x, new_y

def main():
    # --- INITIALIZATION ---
    print("Loading AI model...")
    model = YOLO(MODEL_PATH)

    print("Connecting to Printer...")
    ender3 = p3d.Ender3(PRINTER_PORT)
    ender3.connect()

    print(f"Moving to initial search position...")
    ender3.go_to(x=START_X, y=START_Y, z=Z_HEIGHT)
    
    # Track coordinates in mm
    current_pos_x, current_pos_y, current_pos_z = get_current_pos(ender3)
    
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_PORT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    has_aligned = False
    print("Live Feed Active. Commands: 'q' to quit, 'r' to re-align.")

    # --- LIVE VIDEO & ALIGNMENT LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run AI detection
        results = model(frame, conf=0.7, verbose=False)
        
        # Use a clean copy of the frame to draw our custom boxes
        annotated_frame = frame.copy()

        # Only run drawing and alignment math if the AI actually sees a target
        if len(results[0].boxes) > 0:
            # --- DYNAMIC BOX DRAWING ---
            box = results[0].boxes[0].xyxy[0] 
            
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            
            # Calculate LIVE pixel width and height from the AI box
            bw = float(x_max - x_min)
            bh = float(y_max - y_min)

            # Calculate zoom ratio for the 0.3mm offset
            # physical_width_mm = 1.531 
            # current_pixels_per_mm = bw / physical_width_mm
            # Calculate zoom ratio for the 0.3mm offset
            current_pixels_per_mm = 8150.556 / (current_pos_z - 91.58) #91.58 is offset to ground
            offset = int(0.3 * current_pixels_per_mm)

            # BOX 1: THE TRACKER (Follows the socket, drawn in Green)
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 

            # --- ADD THE CERTAINTY SCORE BACK ---
            # 1. Extract the raw confidence number (e.g., 0.954)
            conf = float(results[0].boxes[0].conf[0])
            
            # 2. Format it as a clean percentage (e.g., "95%")
            conf_text = f"{conf * 100:.0f}%"
            
            # 3. Draw the text right above the top-left corner of the blue box
            cv2.putText(annotated_frame, conf_text, (x_min, y_min - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # BOX 2: STATIC EXACT CENTER (Locked at 800, 600, drawn in Blue)
            c_start_x = int(CENTER_X - (bw / 2))
            c_start_y = int(CENTER_Y - (bh / 2))
            c_end_x = int(CENTER_X + (bw / 2))
            c_end_y = int(CENTER_Y + (bh / 2))
            
            cv2.rectangle(annotated_frame, (c_start_x, c_start_y), (c_end_x, c_end_y), (255, 0, 0), 2)

            # BOX 3: STATIC OFFSET (drawn in Red)
            l_start_x = c_start_x - offset
            l_start_y = c_start_y - offset
            l_end_x = c_end_x + offset
            l_end_y = c_end_y + offset

            cv2.rectangle(annotated_frame, (l_start_x, l_start_y), (l_end_x, l_end_y), (0, 0, 255), 2)
            # --- HARDWARE ALIGNMENT LOGIC ---
            if not has_aligned:
                hw_x, hw_y, hw_z = get_current_pos(ender3)
                
                if hw_x is not None:
                    target_x = float((box[0] + box[2]) / 2)
                    target_y = float((box[1] + box[3]) / 2)

                    # Move the printer using real hardware baseline
                    current_pos_x, current_pos_y = align_target(
                        ender3, target_x, target_y, hw_x, hw_y, hw_z
                    )
                    
                    has_aligned = True 
                    print(f"Alignment complete at X:{current_pos_x:.3f}, Y:{current_pos_y:.3f}")
                else:
                    print("[ERROR] Could not sync with printer position. Retrying...")

        # Always draw the coordinate text (even if target is temporarily lost)
        cv2.putText(annotated_frame, f"X: {current_pos_x:.2f} Y: {current_pos_y:.2f}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Live AI Alignment", annotated_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'): 
            has_aligned = False
            print("\n[RESET] Fetching hardware position for next alignment...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()