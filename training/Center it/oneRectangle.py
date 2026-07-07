import cv2
import os

# --- Configuration ---
RECT_WIDTH = 132
RECT_HEIGHT = 682

def get_target_pixel(image_path):
    """
    Opens the image, allows the user to click to place a rectangle,
    saves the image on 'c', and returns the (x, y) coordinates of the click.
    """
    # Ensure our save folder exists
    os.makedirs("rectangleImages", exist_ok=True)
    
    # Local state dictionary to avoid global variable conflicts
    state = {'x': -1, 'y': -1, 'img_display': None}
    
    img_clean = cv2.imread(image_path)
    if img_clean is None:
        print(f"Error: Could not load '{image_path}'")
        return None, None

    state['img_display'] = img_clean.copy()

    def draw_rectangle_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['x'], state['y'] = x, y
            
            temp_img = img_clean.copy()
            bottom_right = (x + RECT_WIDTH, y + RECT_HEIGHT)
            center = (x + RECT_WIDTH // 2, y + RECT_HEIGHT // 2)
            
            cv2.rectangle(temp_img, (x, y), bottom_right, (255, 0, 0), 2)
            cv2.circle(temp_img, center, 4, (0, 0, 255), -1)
            
            state['img_display'] = temp_img
            cv2.imshow("Target Selector", state['img_display'])

    cv2.namedWindow("Target Selector")
    cv2.setMouseCallback("Target Selector", draw_rectangle_callback)
    
    print(f"\n--- Processing {image_path} ---")
    print("1. Click the target top-left corner.")
    print("2. Press 'c' to confirm and proceed.")
    print("3. Press 'q' to abort.")

    cv2.imshow("Target Selector", state['img_display'])

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27: 
            cv2.destroyAllWindows()
            return None, None
            
        elif key == ord('c'):
            if state['x'] != -1 and state['y'] != -1:
                base_name = os.path.basename(image_path)
                new_filename = f"R_{state['x']}_{state['y']}_{base_name}"
                
                # Save into the rectangleImages folder
                save_path = os.path.join("rectangleImages", new_filename)
                cv2.imwrite(save_path, state['img_display'])
                
                print(f"Target confirmed! Saved as '{save_path}'")
                cv2.destroyAllWindows()
                return state['x'], state['y']
            else:
                print("Wait! You need to click somewhere to draw the rectangle first.")