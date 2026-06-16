import cv2
import time
import py3DCal as p3d
import take_picture as tp

# ==========================================
# 1. Connect to Printer and Take Picture
# ==========================================

# Connect to the Ender3
ender3 = p3d.Ender3("/dev/tty.usbserial-1130")
ender3.connect()

z_height = 110
x_start = 15
y_start = 120
filename = "image_camera_calibration.jpg"

# Move Z to the safe height first
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(z=z_height)

print(f"Moving to X:{x_start}, Y:{y_start}...")
ender3.go_to(x=x_start, y=y_start, z=z_height)

# Wait half a second for the printer frame to stop vibrating
time.sleep(0.5) 

# Take the picture and save it
tp.take_picture(filename, 0)
print(f"Saved: {filename}")

# ==========================================
# 2. Image Selection (Top and Bottom Points)
# ==========================================

# List to store the (x, y) coordinates of the clicks
clicked_points = []

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only accept the first two clicks
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            
            # Determine if this is the top (1st click) or bottom (2nd click)
            point_label = "Top (x1)" if len(clicked_points) == 1 else "Bottom (x2)"
            print(f"-> Registered {point_label} at X:{x}, Y:{y}")
            
            # Draw a green circle and label the point for visual feedback
            cv2.circle(param['img'], (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(param['img'], point_label, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(param['winname'], param['img'])
            
            if len(clicked_points) == 2:
                print("\nBoth points registered! Press ANY KEY on your keyboard to calculate.")

# Load the image
img = cv2.imread(filename)

if img is None:
    print(f"Error: Could not load {filename}. Exiting.")
else:
    winname = "Click TOP point, then BOTTOM point"
    
    cv2.imshow(winname, img)
    cv2.setMouseCallback(winname, select_point, {'img': img, 'winname': winname})
    
    print("\n--- Interaction Required ---")
    print("1. Click the TOP most point on the line (x1).")
    print("2. Click the BOTTOM most point on the line (x2).")
    print("3. Press ANY KEY on your keyboard to proceed.")
    
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ==========================================
    # 3. Calculate Difference & Rotation Logic
    # ==========================================
    
    if len(clicked_points) == 2:
        # Extract just the X values from the clicked (x, y) tuples
        x1 = clicked_points[0][0]  # Top point X
        x2 = clicked_points[1][0]  # Bottom point X
        
        x_diff = x2 - x1
        
        print("\n==============================")
        print("          RESULTS")
        print("==============================")
        print(f"Top Point X-value (x1):    {x1} px")
        print(f"Bottom Point X-value (x2): {x2} px")
        print(f"Difference (x2 - x1):      {x_diff} px\n")
        
        # Apply the rotation correction logic based on X values
        if x_diff > 0:
            print("Status: The X value on the bottom is GREATER (line slopes down and right).")
            print("Action: --> ROTATE CLOCKWISE <--")
        elif x_diff < 0:
            print("Status: The X value on the bottom is LESS (line slopes down and left).")
            print("Action: --> ROTATE COUNTERCLOCKWISE <--")
        else:
            print("Status: The difference is ZERO.")
            print("Action: --> PERFECTLY VERTICAL. No rotation needed. <--")
            
    else:
        print("\nError: Did not get exactly two clicks. Please run the script again and click twice.")