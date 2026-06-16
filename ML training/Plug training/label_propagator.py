import os
import re
import cv2
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "pinBack"  
VIS_DIR = "pinBack_Visuals"
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

def write_label(txt_path, class_id, points, img_w, img_h):
    """Normalizes pixel coordinates and writes them to a YOLOv8-OBB txt file."""
    norm_points = []
    for px, py in points:
        nx = max(0.0, min(1.0, px / img_w))
        ny = max(0.0, min(1.0, py / img_h))
        norm_points.append(f"{nx:.6f} {ny:.6f}")
        
    label_line = f"0 " + " ".join(norm_points)
    with open(txt_path, 'w') as f:
        f.write(label_line + '\n')

def get_base_image(images):
    """Lets the user pick which image they are labeling."""
    print("\nImages found in folder:")
    for i, img in enumerate(images):
        print(f"  [{i}] {img}")
    while True:
        try:
            idx = int(input("\nEnter the number of the image you are labeling: "))
            if 0 <= idx < len(images):
                return images[idx]
            print("Invalid number, try again.")
        except ValueError:
            print("Please enter a number.")

def get_corners_from_user(base_img_path):
    """
    Asks the user to manually type in the 4 corner pixel coordinates.
    Displays the image dimensions so the user knows the valid range.
    """
    img = cv2.imread(base_img_path)
    img_h, img_w = img.shape[:2]

    print(f"\nImage size: {img_w} x {img_h} pixels (width x height)")
    print("Enter the 4 corner coordinates of the bounding box in pixels.")
    print("Go clockwise starting from the TOP-LEFT corner.\n")

    corners = []
    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    for name in corner_names:
        while True:
            try:
                raw = input(f"  {name} (x y): ")
                x, y = map(float, raw.strip().split())
                if 0 <= x <= img_w and 0 <= y <= img_h:
                    corners.append((x, y))
                    break
                else:
                    print(f"  Out of range. x must be 0-{img_w}, y must be 0-{img_h}. Try again.")
            except ValueError:
                print("  Enter two numbers separated by a space, e.g: 120 340")

    return corners, img_w, img_h

def preview_label(base_img_path, corners):
    """Draws the box on the base image and saves a preview so you can verify."""
    img = cv2.imread(base_img_path)
    pts = np.array(corners, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)

    preview_path = os.path.join(VIS_DIR, "_BASE_PREVIEW_" + os.path.basename(base_img_path))
    cv2.imwrite(preview_path, img)
    print(f"\nPreview saved to: {preview_path}")
    print("Open that image to confirm the box looks correct before continuing.")

def main():
    print("--- YOLOv8-OBB Grid Interpolator & Visualizer ---")

    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)

    files = os.listdir(DATA_DIR)
    images = sorted([f for f in files if f.endswith('.jpg') or f.endswith('.png')])

    if not images:
        print(f"Error: No images found in '{DATA_DIR}'.")
        return

    # Step 1: user picks which image they labeled
    base_img_filename = get_base_image(images)
    base_img_path = os.path.join(DATA_DIR, base_img_filename)

    base_data = parse_filename(base_img_filename)
    if not base_data:
        print(f"Error: Filename {base_img_filename} does not match expected format.")
        return

    # Step 2: user types in the 4 corners
    base_points, img_w, img_h = get_corners_from_user(base_img_path)

    # Step 3: save a preview of the base label so user can verify
    preview_label(base_img_path, base_points)
    confirm = input("\nDoes the preview look correct? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Exiting. Re-run the script and enter the correct coordinates.")
        return

    base_x_mm = base_data['x']
    base_y_mm = base_data['y']
    base_z_mm = base_data['z']
    mm_per_pixel = get_mm_per_pixel(base_z_mm)

    print(f"\nBase Image: {base_img_filename}")
    print(f"Base Physical Coords: X:{base_x_mm}, Y:{base_y_mm}, Z:{base_z_mm}")
    print(f"Calculated mm/pixel ratio: {mm_per_pixel:.6f}")

    # Also write the label for the base image itself
    base_txt_path = os.path.join(DATA_DIR, base_img_filename.rsplit('.', 1)[0] + '.txt')
    write_label(base_txt_path, 0, base_points, img_w, img_h)

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

        # Write the new text label
        new_txt_filename = img_file.rsplit('.', 1)[0] + '.txt'
        new_txt_path = os.path.join(DATA_DIR, new_txt_filename)
        write_label(new_txt_path, 0, new_points, img_w, img_h)

        # Draw and save the visualization
        current_img_path = os.path.join(DATA_DIR, img_file)
        draw_img = cv2.imread(current_img_path)
        pts = np.array(new_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        vis_save_path = os.path.join(VIS_DIR, img_file)
        cv2.imwrite(vis_save_path, draw_img)

        generated_count += 1

    print(f"\nDone! Generated {generated_count} labels and saved verification images to '{VIS_DIR}'.")

if __name__ == "__main__":
    main()
