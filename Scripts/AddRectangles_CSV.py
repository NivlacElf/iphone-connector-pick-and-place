import cv2
import csv
import os
import glob
import re

# --- 1. Configuration & Calibration ---
# The exact starting position from your first photo
base_x_mm = 14.0
base_y_mm = 105.0

# The pixel coordinates of the socket in that first photo
base_px_x = 81
base_px_y = 562

# Rectangle dimensions
box_w = 299
box_h = 629

# Movement scale (1mm physical move = 84 pixels screen move)
# Note: If the box moves the opposite direction of the socket, change this to positive 84
pixels_per_mm_x = -70.1
pixels_per_mm_y = 70.1

# --- 2. Setup CSV ---
csv_filename = "pinBack/Socket_Coordinates.csv"
image_folder = "pinBack" # Make sure this matches your folder name!

print("Scanning for images and starting processing...")

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['pixel_x', 'pixel_y', 'mmx', 'mmy', 'filename'])
    
    # --- 3. Dynamically find all images ---
    # This looks for any file matching your naming scheme (e.g. img_x113.0_y119.5_z185.jpg)
    search_pattern = os.path.join(image_folder, "img_x*_y*_z132.jpg")
    image_files = glob.glob(search_pattern)
    
    if not image_files:
        print(f"No images found in the '{image_folder}' directory!")
        
    for filename in image_files:
        # --- 4. Extract X and Y from the filename using Regular Expressions ---
        match = re.search(r"img_x([-0-9.]+)_y([-0-9.]+)_z", filename)
        if not match:
            continue
            
        current_x_mm = float(match.group(1))
        current_y_mm = float(match.group(2))
        
        # --- 5. Calculate new pixel coordinates ---
        # How far did the physical camera move from the start?
        delta_x_mm = current_x_mm - base_x_mm
        delta_y_mm = current_y_mm - base_y_mm
        
        # Apply the 84 pixel/mm ratio to get the new box location
        current_px_x = int(base_px_x + (delta_x_mm * pixels_per_mm_x))
        current_px_y = int(base_px_y + (delta_y_mm * pixels_per_mm_y))
        
        # --- 6. Draw and Save ---
        img = cv2.imread(filename)
        if img is not None:
            start_point = (current_px_x, current_px_y)
            end_point = (current_px_x + box_w, current_px_y + box_h)
            
            # Draw the green rectangle
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            
            # Overwrite the image file
            cv2.imwrite(filename, img)
            
            # Write data to CSV
            writer.writerow([current_px_x, current_px_y, current_x_mm, current_y_mm, filename])
            print(f"Processed {filename} -> Box at ({current_px_x}, {current_px_y})")
        else:
            print(f"Failed to read image: {filename}")

print(f"\nDone! Coordinates saved to {csv_filename}")