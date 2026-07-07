import cv2
import time  # Imported to add a brief pause for camera focus/vibration
import py3DCal as p3d

# --- 1. Connect to Printer ---
ender3 = p3d.Ender3("/dev/tty.usbserial-1130")
ender3.connect()
ender3.initialize(xy_only=False)

z_height = 250
x=201
y=194
# Move Z to the safe height first to avoid crashing into anything during XY travel
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(x, y, z_height)