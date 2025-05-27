import cv2
import numpy as np
import socket
import struct
import pickle

class VideoStreamClient:
    def __init__(self, host='192.168.0.101', port=65432):
        # Socket setup
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        
        print(f"[CLIENT] Connected to {host}:{port}")
    
    def receive_stream(self):
        data = b""
        payload_size = struct.calcsize("L")
        
        try:
            while True:
                # Retrieve message size
                while len(data) < payload_size:
                    data += self.client_socket.recv(4096)
                
                # Extract frame size
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Retrieve frame
                while len(data) < msg_size:
                    data += self.client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize frame
                frame = pickle.loads(frame_data)
                
                # Display frame
                cv2.imshow('Video Stream', frame)
                
                # Break loop if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except Exception as e:
            print(f"[ERROR] {e}")
        
        finally:
            cv2.destroyAllWindows()
            self.client_socket.close()

# Example Usage
def run_client():
    client = VideoStreamClient()
    client.receive_stream()

if __name__ == "__main__":
    run_client()  # Run this on receiving computer
