import cv2
import numpy as np
import math
import os

# --- CONFIGURATION ---
KNOWN_HEIGHT = 775  # The fixed pixel length of your straight plug
CLASS_ID = 0        # The YOLO class index for your plug
IMAGE_PATH = 'test.jpg' # Replace with your image file
# ---------------------

# Global variables
points = []
image = None
clone = None
img_h, img_w = 0, 0

def calculate_rectangle(p1, p2, height):
    """Calculates the four corners given two points and a fixed height."""
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    width = math.hypot(dx, dy)
    
    if width == 0:
        return None
        
    # Perpendicular unit vector pointing "down" relative to the line segment
    ux = -dy / width
    uy = dx / width
    
    x3 = int(x2 + height * ux)
    y3 = int(y2 + height * uy)
    x4 = int(x1 + height * ux)
    y4 = int(y1 + height * uy)
    
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def calculate_angle(p1, p2):
    """Calculates the standard geometric angle of the top line."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Invert dy because OpenCV's Y-axis goes down, but standard math Y-axis goes up
    angle_rad = math.atan2(-dy, dx)
    return math.degrees(angle_rad)

def save_yolo_obb(rect_points, img_w, img_h, image_path, angle):
    """Saves the four corners as normalized coordinates for YOLOv8-OBB."""
    norm_points = []
    for x, y in rect_points:
        norm_points.append(f"{x / img_w:.6f} {y / img_h:.6f}")
        
    label_line = f"{CLASS_ID} " + " ".join(norm_points)
    txt_filename = os.path.splitext(image_path)[0] + '.txt'
    
    with open(txt_filename, 'a') as f:
        f.write(label_line + '\n')
        
    print(f"Saved OBB label to {txt_filename} | Angle: {angle:.2f}°")

def mouse_callback(event, x, y, flags, param):
    global points, clone, image, img_w, img_h
    
    # 1. Handle the clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        # If it's the second click, finalize and save the box
        if len(points) == 2:
            rect_points = calculate_rectangle(points[0], points[1], KNOWN_HEIGHT)
            print(f"x1y1: ({points[0][0]}, {points[0][1]}) | x2y2: ({points[1][0]}, {points[1][1]})")
            if rect_points:
                angle = calculate_angle(points[0], points[1])
                
                # Draw permanent box on the base image
                cv2.line(image, rect_points[0], rect_points[1], (0, 255, 0), 2)
                cv2.line(image, rect_points[1], rect_points[2], (0, 255, 0), 2)
                cv2.line(image, rect_points[2], rect_points[3], (0, 255, 0), 2)
                cv2.line(image, rect_points[3], rect_points[0], (0, 255, 0), 2)
                
                # Draw permanent angle text
                cv2.putText(image, f"{angle:.1f} deg", (points[1][0] + 10, points[1][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                save_yolo_obb(rect_points, img_w, img_h, IMAGE_PATH, angle)
            
            # Reset points for the next object/bounding box if needed
            points = []
            clone = image.copy()
            
        cv2.imshow("Smart Annotator", clone)

    # 2. Handle the live preview move event
    elif event == cv2.EVENT_MOUSEMOVE:
        # Only show preview if we have clicked the first point but not the second
        if len(points) == 1:
            # Refresh the canvas to wipe the previous frame's preview line
            clone = image.copy()
            
            # Draw a small indicator at the first click position
            cv2.circle(clone, points[0], 4, (0, 0, 225), -1)
            
            # Calculate what the box *would* look like at the current mouse position (x, y)
            preview_points = calculate_rectangle(points[0], (x, y), KNOWN_HEIGHT)
            
            if preview_points:
                # Draw the preview box
                cv2.line(clone, preview_points[0], preview_points[1], (0, 225, 0), 1)
                cv2.line(clone, preview_points[1], preview_points[2], (0, 225, 0), 1)
                cv2.line(clone, preview_points[2], preview_points[3], (0, 225, 0), 1)
                cv2.line(clone, preview_points[3], preview_points[0], (0, 225, 0), 1)
                
                # Calculate and display live angle preview next to cursor
                angle = calculate_angle(points[0], (x, y))
                cv2.putText(clone, f"{angle:.1f} deg", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            cv2.imshow("Smart Annotator", clone)

# --- MAIN EXECUTION ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not load {IMAGE_PATH}.")
    exit()

img_h, img_w = image.shape[:2]
clone = image.copy()

cv2.namedWindow("Smart Annotator")
cv2.setMouseCallback("Smart Annotator", mouse_callback)

print(f"--- YOLOv8-OBB Live Preview Annotator ---")
print(f"1. Click the TOP LEFT corner of the plug.")
print(f"2. Move your mouse to preview, then click the TOP RIGHT corner.")
print(f"Press 'r' to reset, 'q' to quit.")

while True:
    cv2.imshow("Smart Annotator", clone)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):
        image = cv2.imread(IMAGE_PATH)  # Reload original clean image
        clone = image.copy()
        points = []
        print("Resetting canvas.")
        
    elif key == ord("q"):
        break

cv2.destroyAllWindows()