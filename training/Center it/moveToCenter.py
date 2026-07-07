import time
import py3DCal as p3d

# --- 1. Configuration & Constants ---
# Printer safety limits
Z_HEIGHT = 185
LIMIT_MAX = 150.0
LIMIT_MIN = 0.0

# Image and rectangle dimensions (pixels)
IMG_W = 1600
IMG_H = 1200
RECT_W = 132
RECT_H = 682

# Calibration ratios
PX_PER_MM_X = -86.5
PX_PER_MM_Y = 86.5

# Calculate the exact center of the camera's view
IMG_CX = IMG_W / 2.0
IMG_CY = IMG_H / 2.0


def align_lens_to_pixel(printer, current_x, current_y, px_x, px_y):
    """
    Calculates alignment offsets and moves the printer so the camera
    is centered on the specified pixel bounding box.
    
    Returns the new (x, y) coordinates of the printer.
    """
    print("\nCalculating alignment moves...")

    # 1. Calculate the center of the bounding box
    rect_cx = px_x + (RECT_W / 2.0)
    rect_cy = px_y + (RECT_H / 2.0)

    # 2. Calculate how far off-center the box is in pixels
    delta_px_x = rect_cx - IMG_CX
    delta_px_y = rect_cy - IMG_CY

    # 3. Convert the pixel offset to physical millimeter movement
    move_mm_x = -delta_px_x / PX_PER_MM_X
    move_mm_y = -delta_px_y / PX_PER_MM_Y

    # 4. Calculate the raw target coordinates
    raw_target_x = current_x + move_mm_x
    raw_target_y = current_y + move_mm_y

    # 5. FAILSAFE: Clamp values between 0 and 150 so it never crashes
    target_x = max(LIMIT_MIN, min(LIMIT_MAX, raw_target_x))
    target_y = max(LIMIT_MIN, min(LIMIT_MAX, raw_target_y))

    # --- Execute the Move ---
    print("Executing move:")
    print(f"  Current POS  : X:{current_x:.2f}, Y:{current_y:.2f}")
    print(f"  Target Raw   : X:{raw_target_x:.3f}, Y:{raw_target_y:.3f}")
    print(f"  Pixel offset : ΔX:{delta_px_x:.2f}px, ΔY:{delta_px_y:.2f}px")

    # Notify the terminal if the failsafe had to kick in
    if raw_target_x != target_x or raw_target_y != target_y:
        print(f"  ⚠️ Failsafe active! Clamped to: X:{target_x:.3f}, Y:{target_y:.3f}")

    printer.go_to(x=target_x, y=target_y, z=Z_HEIGHT)
    time.sleep(0.5) 

    print("Alignment routine complete!")
    
    # Return the new position so your main script can track where the lens currently is
    return target_x, target_y
