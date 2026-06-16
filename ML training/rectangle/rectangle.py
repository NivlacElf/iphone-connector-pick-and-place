import cv2

# --- 1. Global Variables ---
drawing = False  
ix, iy = -1, -1  

# --- 2. The Mouse Callback Function ---
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image, display_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            # First click: Start drawing
            drawing = True
            ix, iy = x, y
        else:
            # Second click: Finish drawing
            drawing = False
            
            # Draw the final rectangle
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            display_image = image.copy()
            
            # --- NEW: Calculate and Print the Data ---
            # Find the true top-left corner (origin)
            origin_x = min(ix, x)
            origin_y = min(iy, y)
            
            # Calculate width and height
            width = abs(x - ix)
            height = abs(y - iy)
            
            print("--- New Rectangle ---")
            print(f"Origin (X, Y): ({origin_x}, {origin_y})")
            print(f"Width:  {width} px")
            print(f"Height: {height} px\n")
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            display_image = image.copy()
            cv2.rectangle(display_image, (ix, iy), (x, y), (0, 255, 0), 2)

# --- 3. Main Setup ---
image = cv2.imread('test.jpg')
display_image = image.copy()

cv2.namedWindow('Image Viewer')
cv2.setMouseCallback('Image Viewer', draw_rectangle)

# --- 4. The Display Loop ---
print("Click once to start. Move the mouse. Click again to finish. Press 'q' or 'Esc' to quit.")

while True:
    cv2.imshow('Image Viewer', display_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: 
        break

cv2.destroyAllWindows()