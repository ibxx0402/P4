import cv2
import socket
import numpy as np
import time 

host = "192.168.0.100"
client = "192.168.0.101"
port = 5000
max_length = 65540

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

# Pre-allocate buffer for efficiency
buffer = bytearray(max_length * 1000)  # Allocate a large buffer once
frame_info = None
frame = None

# Initialize variables for calculating FPS
prev_time = time.time()
fps = 0

while True:
    try:
        # Receive header with timeout
        header = sock.recv(3)

        nums_of_packs = int.from_bytes(header, byteorder="big")
        
        # Reset buffer view for this frame
        buffer_view = memoryview(buffer)
        current_position = 0
        
        # Receive all packets directly into the pre-allocated buffer
        for i in range(nums_of_packs):
            packet_size = sock.recv_into(buffer_view[current_position:current_position+max_length])
            current_position += packet_size
            
        # Create numpy array from buffer slice (no copying)
        frame_data = np.frombuffer(buffer_view[:current_position], dtype=np.uint8)
        
        # Use IMREAD_REDUCED_COLOR_4 for faster decoding when possible
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Use more efficient text rendering by pre-formatting the string
        fps_text = f"FPS: {fps:.1f}"  # Reduce decimal precision for faster formatting
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Stream", frame)
            
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print(f"Error: {e}")
        continue
                    
sock.close()
cv2.destroyAllWindows()
