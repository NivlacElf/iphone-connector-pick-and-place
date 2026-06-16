import cv2
import time  # Imported to add a brief pause for camera focus/vibration
import py3DCal as p3d

# --- 1. Connect to Printer ---
ender3 = p3d.Ender3("/dev/tty.usbserial-140")
ender3.connect()
#ender3.initialize(xy_only=False)

z_height = 185
x=113
y=117
# Move Z to the safe height first to avoid crashing into anything during XY travel
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(x, y, z_height)