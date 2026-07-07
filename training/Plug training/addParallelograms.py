import os
import re
import cv2
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "pinBack"  
VIS_DIR = "pinBack_Visuals" # New folder for visual verification
# ---------------------

def get_mm_per_pixel(z):
    """Calculates the mm/pixel ratio based on the Z height."""
    return (z - 125.0) / 8150.556

def parse_filename(filename):
    """Extracts degree, x, y, and z from the filename."""
    match = re.match(r"^(.*)_img_x([-\d.]+)_y([-\d.]+)_z([-\d.]+)\.(jpg|png)$", filename)
    if match:
        return {
            "degree": match.group(1),
            "x": float(match.group(2)),
            "y": float(match.group(3)),
            "z": float(match.group(4))
        }
    return None

def read_base_label(txt_path, img_w, img_h):
    """Reads the normalized YOLOv8-OBB txt file and returns pixel coordinates."""
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
    
    parts = line.split()
    class_id = int(parts[0])
    
    points = []
    for i in range(1, 9, 2):
        px = float(parts[i]) * img_w
        py = float(parts[i+1]) * img_h
        points.append((px, py))
        
    return class_id, points

def write_label(txt_path, class_id, points, img_w, img_h):
    """Normalizes pixel coordinates and writes them to a YOLOv8-OBB txt file."""
    norm_points = []
    for px, py in points:
        nx = max(0.0, min(1.0, px / img_w))
        ny = max(0.0, min(1.0, py / img_h))
        norm_points.append(f"{nx:.6f} {ny:.6f}")
        
    label_line = f"{class_id} " + " ".join(norm_points)
    with open(txt_path, 'w') as f:
        f.write(label_line + '\n')

def main():
    print("--- YOLOv8-OBB Grid Interpolator & Visualizer ---")
    
    # Create the visualization directory if it doesn't exist
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
    
    files = os.listdir(DATA_DIR)
    images = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    labels = [f for f in files if f.endswith('.txt')]
    
    if len(labels) == 0:
        print("Error: No .txt label found. Please label ONE image using the annotator script first.")
        return
    if len(labels) > 1:
        print("Error: Multiple .txt files found. Leave only your ONE base label in the directory.")
        return
        
    base_txt_filename = labels[0]
    base_img_filename = base_txt_filename.replace('.txt', '.jpg')
    
    if base_img_filename not in images:
        print(f"Error: Could not find the image {base_img_filename} matching your label.")
        return

    base_data = parse_filename(base_img_filename)
    if not base_data:
        print(f"Error: Base image filename {base_img_filename} does not match expected format.")
        return
        
    base_img_path = os.path.join(DATA_DIR, base_img_filename)
    img = cv2.imread(base_img_path)
    img_h, img_w = img.shape[:2]
    
    class_id, base_points = read_base_label(os.path.join(DATA_DIR, base_txt_filename), img_w, img_h)
    
    base_x_mm = base_data['x']
    base_y_mm = base_data['y']
    base_z_mm = base_data['z']
    mm_per_pixel = get_mm_per_pixel(base_z_mm)
    
    print(f"Base Image: {base_img_filename}")
    print(f"Base Physical Coords: X:{base_x_mm}, Y:{base_y_mm}, Z:{base_z_mm}")
    print(f"Calculated mm/pixel ratio: {mm_per_pixel:.6f}")
    
    generated_count = 0
    for img_file in images:
        if img_file == base_img_filename:
            continue 
            
        current_data = parse_filename(img_file)
        if not current_data:
            continue
            
        # Physical deltas
        delta_x_mm = current_data['x'] - base_x_mm
        delta_y_mm = current_data['y'] - base_y_mm
        
        # Pixel shifts
        shift_x_pixels = -(delta_x_mm / mm_per_pixel)
        shift_y_pixels = -(delta_y_mm / mm_per_pixel)
        
        # Apply shifts to all 4 corners
        new_points = []
        for px, py in base_points:
            new_px = px + shift_x_pixels
            new_py = py + shift_y_pixels
            new_points.append((new_px, new_py))
            
        # 1. Write the new text label
        new_txt_filename = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        new_txt_path = os.path.join(DATA_DIR, new_txt_filename)
        write_label(new_txt_path, class_id, new_points, img_w, img_h)
        
        # 2. Draw and save the visualization
        current_img_path = os.path.join(DATA_DIR, img_file)
        draw_img = cv2.imread(current_img_path)
        
        # Convert points to a numpy array format that OpenCV polylines expects
        pts = np.array(new_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw the box in green, 2 pixels thick
        cv2.polylines(draw_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Save the visualization image
        vis_save_path = os.path.join(VIS_DIR, img_file)
        cv2.imwrite(vis_save_path, draw_img)
        
        generated_count += 1
        
    print(f"Successfully generated {generated_count} labels and saved verification images to '{VIS_DIR}' folder!")

if __name__ == "__main__":
    main()