import cv2
import os
import time
import py3DCal as p3d

# Import your custom modules
from take_picture import take_picture
# We import the dimensions directly from moveToCenter to ensure the auto-drawing matches the math perfectly
from moveToCenter import align_lens_to_pixel, Z_HEIGHT, IMG_W, IMG_H, RECT_W, RECT_H
from oneRectangle import get_target_pixel

def main():
    # 1. Setup & Connection
    os.makedirs("rectangleImages", exist_ok=True)
    
    print("Connecting to printer...")
    ender3 = p3d.Ender3("/dev/tty.usbserial-140")
    ender3.connect()

    print(f"Moving to safe Z height: {Z_HEIGHT}...")
    ender3.go_to(z=Z_HEIGHT)
    time.sleep(1)

    # Initial start coordinates
    current_x = 110.0
    current_y = 116.0
    
    print(f"Moving to start position X:{current_x}, Y:{current_y}...")
    ender3.go_to(x=current_x, y=current_y, z=Z_HEIGHT)
    time.sleep(1)

    # --- First Pass: Capture & Manually Target ---
    filename_1 = f"img_x{current_x}_y{current_y}_z{Z_HEIGHT}.jpg"
    take_picture(filename_1)
    
    # Opens UI to click the target (This saves the first image to rectangleImages)
    px_x, px_y = get_target_pixel(filename_1)
    
    if px_x is None or px_y is None:
        print("Operation aborted by user.")
        return

    # --- Alignment: Calculate & Move ---
    current_x, current_y = align_lens_to_pixel(
        printer=ender3,
        current_x=current_x,
        current_y=current_y,
        px_x=px_x,
        px_y=px_y
    )
    time.sleep(1) 

    # --- Second Pass: Auto-draw Centered Target ---
    print("\n--- Verification Pass (Auto-Centering) ---")
    filename_2 = f"img_centered_x{current_x:.2f}_y{current_y:.2f}_z{Z_HEIGHT}.jpg"
    take_picture(filename_2)
    
    # Load the new picture
    img_centered = cv2.imread(filename_2)
    
    if img_centered is not None:
        # Calculate the exact center of the image
        center_x = int(IMG_W // 2)
        center_y = int(IMG_H // 2)
        
        # Calculate where the top-left of the rectangle should be to perfectly center it
        top_left_x = int(center_x - (RECT_W // 2))
        top_left_y = int(center_y - (RECT_H // 2))
        bottom_right = (int(top_left_x + RECT_W), int(top_left_y + RECT_H))

        # Draw the rectangle and the center dot
        cv2.rectangle(img_centered, (top_left_x, top_left_y), bottom_right, (255, 0, 0), 2)
        cv2.circle(img_centered, (center_x, center_y), 4, (0, 0, 255), -1)

        # Save to the specific folder
        final_path = os.path.join("rectangleImages", f"AutoCentered_{filename_2}")
        cv2.imwrite(final_path, img_centered)
        
        print(f"Verification image automatically drawn and saved to: '{final_path}'")
        print("\nProcess finished successfully!")
    else:
        print(f"Error: Could not read '{filename_2}' to draw the verification rectangle.")

if __name__ == "__main__":
    main()