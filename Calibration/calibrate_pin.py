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

z_height = 104
x_start = 43
y_start = 165
filename = "image_camera_calibration.jpg"

# Move Z to the safe height first
print(f"Moving to safe Z height: {z_height}...")
ender3.go_to(z=z_height)

print(f"Moving to X:{x_start}, Y:{y_start}...")
ender3.go_to(x=x_start, y=y_start, z=z_height)

# Wait half a second for the printer frame to stop vibrating
time.sleep(0.5) 

# Take the picture and save it
tp.take_picture(filename, 1)
print(f"Saved: {filename}")

# ==========================================
# 2. Image Selection (Left and Right Points)
# ==========================================

# List to store the (x, y) coordinates of the clicks
clicked_points = []

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only accept the first two clicks
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            
            # Determine if this is the left (1st click) or right (2nd click)
            point_label = "Left (y1)" if len(clicked_points) == 1 else "Right (y2)"
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
    winname = "Click LEFT point, then RIGHT point"
    
    cv2.imshow(winname, img)
    cv2.setMouseCallback(winname, select_point, {'img': img, 'winname': winname})
    
    print("\n--- Interaction Required ---")
    print("1. Click the LEFT most point on the line (y1).")
    print("2. Click the RIGHT most point on the line (y2).")
    print("3. Press ANY KEY on your keyboard to proceed.")
    
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ==========================================
    # 3. Calculate Difference & Rotation Logic
    # ==========================================
    
    if len(clicked_points) == 2:
        # Extract just the Y values from the clicked (x, y) tuples
        y1 = clicked_points[0][1]  # Left point
        y2 = clicked_points[1][1]  # Right point
        
        y_diff = y2 - y1
        
        print("\n==============================")
        print("          RESULTS")
        print("==============================")
        print(f"Left Point Y-value (y1):   {y1} px")
        print(f"Right Point Y-value (y2):  {y2} px")
        print(f"Difference (y2 - y1):      {y_diff} px\n")
        
        # Apply the rotation correction logic
        # Note: In OpenCV, Y increases as you go DOWN the screen.
        if y_diff > 0:
            print("Status: The Y value INCREASED (line slopes down on the right).")
            print("Action: --> ROTATE COUNTERCLOCKWISE <--")
        elif y_diff < 0:
            print("Status: The Y value DECREASED (line slopes up on the right).")
            print("Action: --> ROTATE CLOCKWISE <--")
        else:
            print("Status: The difference is ZERO.")
            print("Action: --> PERFECTLY ALIGNED. No rotation needed. <--")
            
    else:
        print("\nError: Did not get exactly two clicks. Please run the script again and click twice.")