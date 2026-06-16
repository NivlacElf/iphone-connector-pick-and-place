import cv2
import time  # Imported to add a brief pause for camera focus/vibration
import py3DCal as p3d
import take_picture as tp

# --- 1. Connect to Printer ---
ender3 = p3d.Ender3("/dev/tty.usbserial-1130")
ender3.connect()
# ender3.initialize(xy_only=False)

z_height = 200
x=205
y=199
# Move Z to the safe height first to avoid crashing into anything during XY travel
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(x, y, z_height)
filename = f"{z_height}z.jpg"
time.sleep(8)  # Pause for 2 seconds to allow camera to stabilize
# filename = f"img_x{x}_y{y}_z{z_height}.jpg"
#tp.take_picture(filename)