import os
from PIL import Image

# --- Configuration ---
folder_path = "retrain_images"    # Path to your folder of images
output_name = "socket_grid_6x6.png"
thumb_w, thumb_h = 400, 400       # High-res size for each individual image
cols, rows = 6, 6                 # Strict 6x6 grid layout
max_images = cols * rows          # 36 images total

# --- 1. Load and Grab First 36 Images ---
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()  # Keeps them in alphabetical/numerical order
batch = image_files[:max_images]  # Slice to get exactly the first 36

if len(batch) == 0:
    print(f"Error: No images found in '{folder_path}'!")
    exit()
elif len(batch) < max_images:
    print(f"Warning: Found only {len(batch)} images. Filling the rest of the 6x6 grid with white space.")

# --- 2. Initialize the Canvas ---
canvas_w = cols * thumb_w
canvas_h = rows * thumb_h
canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

# --- 3. Stitch Images onto the Canvas ---
for idx, file_path in enumerate(batch):
    # Calculate grid position (0 to 5 for columns and rows)
    c = idx % cols
    r = idx // cols
    
    # Calculate pixel offsets
    x_offset = c * thumb_w
    y_offset = r * thumb_h
    
    try:
        with Image.open(file_path) as img:
            # Resize and paste
            img_resized = img.resize((thumb_w, thumb_h))
            canvas.paste(img_resized, (x_offset, y_offset))
    except Exception as e:
        print(f"Skipping corrupted file {file_path}: {e}")

# --- 4. Save the Final Grid ---
canvas.save(output_name)
print(f"Success! Perfect 6x6 grid saved as '{output_name}' ({canvas_w}x{canvas_h} px).")