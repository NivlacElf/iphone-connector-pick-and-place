import cv2
from ultralytics import YOLO

def main():
    # 1. Load your custom brain
    print("Loading AI model...")
    model = YOLO('best.pt')
    print("Model loaded successfully!")

    # 2. Open the camera 
    # '0' is usually the built-in webcam. Change to '1' or '2' for the microscope if needed.
    camera_port = 0 
    cap = cv2.VideoCapture(camera_port)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_port}")
        return

    print("Camera active. Press 'q' to quit.")

    # 3. The Live AI Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- The Magic Happens Here ---
        # Feed the frame to the AI. 
        # conf=0.5 means "only draw a box if you are at least 50% sure it's the target"
        results = model(frame, conf=0.5, verbose=False)

        # Tell the AI to draw its bounding boxes directly onto the image frame
        annotated_frame = results[0].plot()

        # Display the live feed with the AI's drawings
        cv2.imshow("AI Vision Test", annotated_frame)

        # Quit if 'q' or ESC is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()