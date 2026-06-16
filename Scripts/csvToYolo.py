import os
import shutil
import csv

# --- 1. Configuration ---
CSV_FILE = 'Socket_Coordinates.csv'

# Your exact column headers from the CSV
COL_FILENAME = 'filename' 
COL_PX_X = 'pixel_x'
COL_PX_Y = 'pixel_y'

# Known Camera & Rectangle Dimensions
IMG_W = 1600.0
IMG_H = 1200.0
RECT_W = 299
RECT_H = 629

def main():
    # 2. Setup the exact folder structure YOLOv8 requires
    os.makedirs("yolo_dataset/images/train", exist_ok=True)
    os.makedirs("yolo_dataset/labels/train", exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find '{CSV_FILE}'")
        return

    success_count = 0
    missing_count = 0

    # 3. Read the CSV and process each image
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # The filename in your CSV includes the folder (e.g., 'ImagesV5_rectangles/...')
            source_image_path = row[COL_FILENAME]
            
            # We want to extract JUST the image name for saving in the new folder
            clean_filename = os.path.basename(source_image_path)
            
            try:
                top_left_x = float(row[COL_PX_X])
                top_left_y = float(row[COL_PX_Y])
            except ValueError:
                print(f"Skipping {clean_filename} - invalid coordinates.")
                continue

            if not os.path.exists(source_image_path):
                print(f"Skipping {clean_filename} - Image not found at '{source_image_path}'.")
                missing_count += 1
                continue

            # 4. Do the YOLO Math (Convert top-left pixels to normalized center percentages)
            center_x = top_left_x + (RECT_W / 2.0)
            center_y = top_left_y + (RECT_H / 2.0)
            
            norm_center_x = center_x / IMG_W
            norm_center_y = center_y / IMG_H
            norm_w = RECT_W / IMG_W
            norm_h = RECT_H / IMG_H

            # 5. Write the YOLO label text file
            txt_filename = clean_filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join("yolo_dataset/labels/train", txt_filename)
            
            # The '0' at the start is the Class ID (0 = your target)
            with open(label_path, "w") as f:
                f.write(f"0 {norm_center_x:.6f} {norm_center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # 6. Copy the clean image to the YOLO dataset folder
            image_dest_path = os.path.join("yolo_dataset/images/train", clean_filename)
            shutil.copy(source_image_path, image_dest_path)
            
            success_count += 1

    # 7. Create the dataset.yaml file that tells YOLO where to look
    yaml_path = os.path.join("yolo_dataset", "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/train\n\n") 
        f.write("nc: 1\n")
        f.write("names: ['target']\n")

    print(f"\n--- DONE ---")
    print(f"Successfully processed: {success_count} images.")
    if missing_count > 0:
        print(f"Missing images: {missing_count} (Check to make sure the 'ImagesV5_rectangles' folder is in the same directory as this script).")
    print("Next step: Right-click the 'yolo_dataset' folder and zip it!")

if __name__ == "__main__":
    main()