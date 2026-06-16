import cv2

# --- 1. The Mouse Callback Function ---
def click_event(event, x, y, flags, param):
    global image
    
    # When you left-click down
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the exact pixel coordinates to the terminal
        print(f"X: {x}, Y: {y}")
        
        # Optional: Draw a tiny red dot (radius 3) where you clicked for visual confirmation
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Image Viewer', image)

# --- 2. Main Setup ---
# Replace this with the path to one of your microscope images
image_path = 'test.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not find image at '{image_path}'")
else:
    # Create the window and attach the click listener
    cv2.namedWindow('Image Viewer')
    cv2.setMouseCallback('Image Viewer', click_event)
    
    # --- 3. The Display Loop ---
    print("Click anywhere on the image to get coordinates. Press 'q' or 'Esc' to quit.")
    
    # Show the initial image
    cv2.imshow('Image Viewer', image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the Esc key
            break
            
    cv2.destroyAllWindows()