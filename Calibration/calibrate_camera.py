import cv2
import time
import py3DCal as p3d
import take_picture as tp

# ==========================================
# 1. Connect to Printer and Take Pictures
# ==========================================

# Connect to the Ender3
ender3 = p3d.Ender3("/dev/tty.usbserial-1130")
ender3.connect()

z_height = 110

# Move Z to the safe height first
# print(f"Moving to safe Z height: {z_height}...")
# ender3.go_to(z=z_height)

# Define the two points you want to visit
points = [
    {"x": 17, "y": 106, "filename": "image_1_x98.jpg"},
    {"x": 3,  "y": 106, "filename": "image_2_x83.jpg"}
]

print("Starting image capture...")
ender3.go_to(z=z_height)  # Ensure we're at the safe Z height before starting
for pt in points:
    print(f"Moving to X:{pt['x']}, Y:{pt['y']}...")
    ender3.go_to(x=pt['x'], y=pt['y'])
    ender3.go_to(z=110)  # Ensure we're at the safe Z height for each capture
    
    # Wait half a second for the printer frame to stop vibrating
    time.sleep(1) 
    
    # Take the picture and save it
    tp.take_picture(pt['filename'], 0)
    print(f"Saved: {pt['filename']}")

print("Image capture complete!")

# ==========================================
# 2. Image Selection and Pixel Calculation
# ==========================================

# List to store the Y-coordinates of the clicks
clicked_points = []

# Mouse callback function to capture coordinates on click
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the y-coordinate
        clicked_points.append(y)
        print(f"-> Clicked at X:{x}, Y:{y}")
        
        # Draw a small green circle where the user clicked for visual feedback
        cv2.circle(param['img'], (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow(param['winname'], param['img'])

# Loop through the saved images and get user clicks
for pt in points:
    filename = pt['filename']
    img = cv2.imread(filename)
    
    if img is None:
        print(f"Error: Could not load {filename}")
        continue

    winname = f"Click target, then press ANY KEY to continue ({filename})"
    
    # Create window and set mouse callback
    cv2.imshow(winname, img)
    cv2.setMouseCallback(winname, select_point, {'img': img, 'winname': winname})
    
    print(f"\nWaiting for click on {filename}...")
    print("Click your target point, then press any key on your keyboard to proceed.")
    
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# 3. Calculate Difference
# ==========================================

if len(clicked_points) >= 2:
    y1 = clicked_points[0]
    y2 = clicked_points[1]
    
    # Calculate difference
    y_diff = y2-y1
    abs_y_diff = abs(y_diff)
    #if the pixel different is positivev the camera is rotated counterclickwise, if positive the camera is rotated clockwise.
    print("\n--- Results ---")
    print(f"Y-coordinate in Image 1: {y1} px")
    print(f"Y-coordinate in Image 2: {y2} px")
    print(f"Raw Difference (Y2-Y1): {y_diff} px")
    print(f"Absolute Y Pixel Difference: {abs_y_diff} px")
    if y_diff > 0:
        print("rotate clockwise")
    elif y_diff < 0:
        print("rotate counterclockwise")
else:
    print("\nError: Did not get enough clicks. Make sure to click once on both images.")