import time
import re
import py3DCal as p3d

# --- CONFIGURATION ---
PRINTER_PORT = "/dev/tty.usbserial-140" # Ensure this matches your port

def get_current_pos_test(ender3):
    print(f"Sending M114 to printer...")
    ender3.send_gcode("M114")
    
    # We will try to read the response multiple times to clear out any "ok" messages
    max_retries = 10
    for i in range(max_retries):
        time.sleep(0.2) 
        response = ender3.get_response()
        
        if not response:
            continue
            
        print(f"Read Attempt {i+1}: {response.strip()}")

        # Look for the coordinate string in the response
        x_match = re.search(r"X:([-+]?[0-9]*\.?[0-9]+)", response)
        y_match = re.search(r"Y:([-+]?[0-9]*\.?[0-9]+)", response)

        if x_match and y_match:
            x_val = float(x_match.group(1))
            y_val = float(y_match.group(1))
            return x_val, y_val
            
    print("Error: Timed out waiting for coordinate string.")
    return None, None

def main():
    print("Connecting to Printer...")
    try:
        ender3 = p3d.Ender3(PRINTER_PORT)
        ender3.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Test 1: Current position
    print("\n--- TEST 1: Get Current Position ---")
    x, y = get_current_pos_test(ender3)
    if x is not None:
        print(f"SUCCESS! Parsed Coordinates -> X: {x}, Y: {y}")

    # Test 2: Move and verify
    print("\n--- TEST 2: Move +5mm and Verify ---")
    if x is not None:
        new_x = x + 5.0
        print(f"Moving to X: {new_x}...")
        ender3.go_to(x=new_x)
        
        # Wait for move to finish
        time.sleep(1.0) 
        
        final_x, final_y = get_current_pos_test(ender3)
        print(f"Verification -> X: {final_x}, Y: {final_y}")

    print("\nTest Complete.")

if __name__ == "__main__":
    main()