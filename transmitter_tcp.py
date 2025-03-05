import cv2
import numpy as np
import socket
import struct
import pickle
from picamera2 import Picamera2
import time

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=65432):
        # Socket setup
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        
        print(f"[SERVER] Listening on {host}:{port}")
        
        # Camera setup
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (1280, 720)}, lores={"size": (320, 240)}, display="lores")
        self.picam2.configure(config)
        
    def start_streaming(self):
        # Wait for client connection
        client_socket, addr = self.server_socket.accept()
        print(f"[CONNECTION] Connected to {addr}")
        
        # Start camera
        self.picam2.start()
        #time.sleep(0.1)
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Optional: Process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Serialize frame
                data = pickle.dumps(frame)
                
                # Send frame size first
                client_socket.sendall(struct.pack("L", len(data)) + data)
        
        except Exception as e:
            print(f"[ERROR] {e}")
        
        finally:
            self.picam2.stop()
            client_socket.close()
            self.server_socket.close()

# Example Usage
def run_server():
    server = VideoStreamServer()
    server.start_streaming()

# Note: Run server.py on Raspberry Pi, client.py on receiving computer
if __name__ == "__main__":
    run_server()  # Run this on Raspberry Pi
