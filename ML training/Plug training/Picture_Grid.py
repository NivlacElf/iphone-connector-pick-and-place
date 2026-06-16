import cv2
import time  # Imported to add a brief pause for camera focus/vibration
import py3DCal as p3d
import take_picture as tp

# --- 1. Connect to Printer ---
ender3 = p3d.Ender3("/dev/tty.usbserial-1130")
ender3.connect()
# ender3.initialize(xy_only=False)

z_height = 200

# Move Z to the safe height first to avoid crashing into anything during XY travel
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(z=z_height)


# --- 2. Define the Grid ---
# Multiply by 10 to allow integer steps (e.g., 113.0 becomes 1130)
# X goes from 113 down to 97 (-5 step)
x_start = 2065
x_end = 1970   # We use 965 so the loop includes exactly 97.0
x_step = -5 

# Y goes from 115 up to 119 (+5 step)
y_start = 1425
y_end = 1390  # We use 1195 so the loop includes exactly 119
y_step = -5 

# --- 3. Execute the Grid Capture ---
print("Starting grid capture...")

for x_val in range(x_start, x_end, x_step):
    for y_val in range(y_start, y_end, y_step):
        
        # Convert the integer back to a decimal (e.g., 1125 becomes 112.5mm)
        x = x_val / 10.0
        y = y_val / 10.0
        
        print(f"Moving to X:{x}, Y:{y}...")
        ender3.go_to(x=x, y=y, z=z_height)
        
        # Wait half a second for the printer frame to stop vibrating
        time.sleep(0.5) 
        
        # Create a dynamic filename based on the current coordinates
        filename = f"-149.9/-149.9_img_x{x}_y{y}_z{z_height}.jpg"
        
        # Take the picture and save it
        img = tp.take_picture(filename)
        print(f"Saved: {filename}")

print("Grid capture complete!")
# Draw rectangle on image
#Origin (X, Y): (586, 284)
# Width:  188
# Height: 742
# img = cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
#save image as images/img_x106_y118_z185.jpg