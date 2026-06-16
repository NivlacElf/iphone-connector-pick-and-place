import cv2
import time

def take_picture(filename, camera_port):
    # 1. Open the camera. 
    # '0' is usually your laptop's built-in webcam. 
    # '1' or '2' will usually be the USB microscope.
    cap = cv2.VideoCapture(camera_port)

    # Check if the computer actually found a camera on that port
    if not cap.isOpened():
        print(f"Error: Could not connect to camera port {camera_port}. Try changing it to 0 or 2.")
        return

    # Give the camera hardware 1 second to adjust its auto-exposure and focus
    print("Camera connected. Adjusting exposure...")
    time.sleep(1)

    # 2. Grab a single frame
    ret, frame = cap.read()

    if ret:
        # 3. Save the image to the same folder this script is in
        cv2.imwrite(filename, frame)
        print(f"Success! Picture saved as '{filename}'")
    else:
        print("Error: Camera connected, but could not capture an image.")

    # 4. Release the camera so it isn't locked up
    cap.release()

    return frame

if __name__ == "__main__":
    take_picture()