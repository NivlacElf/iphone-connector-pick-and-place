import cv2
import time
import threading
import py3DCal as p3d
from ultralytics import YOLO

# Global flag to safely shut down the background thread
is_running = True

def printer_patrol_thread():
    """
    Background thread that sweeps the printer in a zigzag pattern 
    within the bounds specified by the user.
    """
    global is_running
    
    # --- 1. Bounds Configuration ---
    X_MIN, X_MAX = 97.0, 113.0
    Y_MIN, Y_MAX = 112.5, 117.0
    Z_HEIGHT = 185
    
    STEP_SIZE_X = 1.0  # mm per move
    STEP_SIZE_Y = 1.5  # mm shift after each X sweep
    PAUSE = 0.4        # seconds to let camera settle

    print("[Printer] Connecting...")
    try:
        ender3 = p3d.Ender3("/dev/tty.usbserial-140")
        ender3.connect()
        
        print(f"[Printer] Initializing at Z:{Z_HEIGHT}")
        ender3.go_to(z=Z_HEIGHT)
        time.sleep(2)

        curr_x = X_MIN
        curr_y = Y_MIN
        dir_x = 1  # 1 for right, -1 for left

        print(f"[Printer] Starting area sweep: X[{X_MIN}-{X_MAX}], Y[{Y_MIN}-{Y_MAX}]")
        
        while is_running:
            # Move to the calculated position
            ender3.go_to(x=curr_x, y=curr_y, z=Z_HEIGHT)
            time.sleep(PAUSE)

            # Calculate next X position
            curr_x += (STEP_SIZE_X * dir_x)

            # Check if we hit the X boundaries
            if curr_x > X_MAX or curr_x < X_MIN:
                # Reverse X direction
                dir_x *= -1
                # Snap back to boundary to prevent drift
                curr_x = X_MAX if curr_x > X_MAX else X_MIN
                
                # Increment Y to move to the next "row"
                curr_y += STEP_SIZE_Y
                
                # If we finish the Y range, reset to the bottom
                if curr_y > Y_MAX:
                    print("[Printer] Area sweep complete. Resetting to start...")
                    curr_y = Y_MIN
                    curr_x = X_MIN
                    dir_x = 1
                    time.sleep(1) # Brief pause at reset
            
    except Exception as e:
        print(f"[Printer Error] {e}")
    finally:
        print("[Printer] Patrol thread stopped.")


def main():
    global is_running
    
    # 1. Start the printer sweep in the background
    patrol_thread = threading.Thread(target=printer_patrol_thread)
    patrol_thread.start()

    # 2. Load the AI Model
    print("[Vision] Loading AI model 'best.pt'...")
    model = YOLO('best.pt')
    
    # 3. Open the Microscope Camera
    # Try 0, 1, or 2 to find your microscope
    camera_port = 0 
    cap = cv2.VideoCapture(camera_port)

    if not cap.isOpened():
        print(f"[Vision] Error: Could not open camera {camera_port}")
        is_running = False 
        patrol_thread.join()
        return

    print("\n" + "="*45)
    print("AI TRACKING & PRINTER SWEEP ACTIVE")
    print(f"Sweeping X: 97-113 | Y: 112.5-117")
    print("Press 'q' or ESC to stop.")
    print("="*45 + "\n")

    # 4. Live Vision Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run AI Inference
        results = model(frame, conf=0.5, verbose=False)

        # Draw the tracking box
        annotated_frame = results[0].plot()

        # Show the video feed
        cv2.imshow("Microscope AI Tracker", annotated_frame)

        # Exit handler
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            print("Shutting down...")
            break

    # 5. Safe Cleanup
    is_running = False 
    cap.release()
    cv2.destroyAllWindows()
    patrol_thread.join()
    print("All systems stopped.")

if __name__ == "__main__":
    main()