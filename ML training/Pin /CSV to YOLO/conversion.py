import os
import pandas as pd

# --- 1. Configuration ---
csv_file = "Socket_Coordinates.csv"
output_folder = "labels"

# SET YOUR IMAGE DIMENSIONS HERE (e.g., 1920x1080, 1600x1200, etc.)
# Replace these with your actual camera resolution!
IMG_WIDTH = 1600   
IMG_HEIGHT = 1200  

# Given rectangle dimensions
BOX_W = 163
BOX_H = 618
CLASS_ID = 0  # 0 for your 'socket' class

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# --- 2. Load and Process Data ---
df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} rows. Generating YOLO text files...")

for idx, row in df.iterrows():
    # Extract coordinates (top-left corner)
    top_left_x = row['pixel_x']
    top_left_y = row['pixel_y']
    raw_filename = row['filename']
    
    # Get just the base filename without subfolders (e.g., "img_x98.0_y118.0_z185.jpg")
    base_name = os.path.basename(raw_filename)
    # Swap out the image extension for .txt
    txt_name = os.path.splitext(base_name)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_name)
    
    # --- 3. Calculate YOLO Normalized Coordinates ---
    # Find center points
    center_x = top_left_x + (BOX_W / 2.0)
    center_y = top_left_y + (BOX_H / 2.0)
    
    # Normalize by image resolution
    norm_x = center_x / IMG_WIDTH
    norm_y = center_y / IMG_HEIGHT
    norm_w = BOX_W / IMG_WIDTH
    norm_h = BOX_H / IMG_HEIGHT
    
    # --- 4. Write to YOLO Text File ---
    # Format: class_id x_center y_center width height
    yolo_line = f"{CLASS_ID} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
    
    with open(txt_path, "w") as f:
        f.write(yolo_line)

print(f"Done! Created {len(df)} label files inside the '{output_folder}/' directory.")