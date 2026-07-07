import cv2
import threading
import time
from ultralytics import YOLO
import py3DCal as p3d

# --- 1. System Initialization ---
# Load your brand new, perfectly tight model brain
model = YOLO("best.pt") 

# Connect to the printer
ender3 = p3d.Ender3("/dev/tty.usbserial-140")
ender3.connect()

z_height = 185
print("Homing and moving to safe Z height...")
ender3.go_to(z=z_height)

# Define a looping path (a box pattern around your target area)
motion_path = [
    (113.0, 114.5),
    (97.0, 114.5),
    (97.0, 111.0),
    (113.0, 111.0),
    (113.0, 114.5) # Return to start
]

# Control flag to sync the camera stream and the printer thread
system_active = True


# --- 2. Background Motion Thread ---
def printer_control_worker():
    """Handles printer movements in the background so the camera feed doesn't freeze."""
    global system_active
    print("Background motion thread started.")
    time.sleep(2)  # Give the camera a moment to warm up
    
    while system_active:
        for x, y in motion_path:
            if not system_active:
                break
            print(f"[PRINTER] Moving to X: {x}, Y: {y}")
            ender3.go_to(x=x, y=y, z=z_height)
            
            # Brief pause at the corner to let vibrations settle
            time.sleep(0.8) 
            
        # If you only want to loop through the path once, uncomment the line below:
        # system_active = False 

# Fire up the printer thread
motion_thread = threading.Thread(target=printer_control_worker)
motion_thread.daemon = True # Allows script to close cleanly if main loop exits
motion_thread.start()


# --- 3. Main Live Video & Tracking Loop ---
# Open your camera stream (adjust the index if using an external USB microscope, usually 1 or 2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam/microscope stream.")
    system_active = False

print("\n--- Live Tracking Active ---")
print("Press 'q' in the video window to shut down cleanly.")

while cap.isOpened() and system_active:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Run the image frame through YOLO
    # stream=True uses a generator for optimal real-time memory management
    results = model(frame, stream=True)

    # Annotate the frame with your custom tight bounding box
    annotated_frame = frame.copy()
    for r in results:
        annotated_frame = r.plot() # Automatically handles drawing the rectangle and label

    # Display the tracking results
    cv2.imshow("Real-Time YOLO Socket Tracking", annotated_frame)

    # Break out cleanly if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Shutdown command received.")
        system_active = False
        break

# --- 4. Clean Shutdown ---
print("Stopping threads and closing hardware connections...")
system_active = False
motion_thread.join(timeout=2)
cap.release()
cv2.destroyAllWindows()
print("System disconnected cleanly.")