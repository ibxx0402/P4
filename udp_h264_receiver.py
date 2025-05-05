import socket
import numpy as np
import cv2
import av
import io
import time

def main():
    # UDP socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('0.0.0.0', 9999)  # Listen on all interfaces, port 5000
    sock.bind(server_address)
    sock.settimeout(1.0)  # Add timeout to allow keyboard interrupt
    
    print(f"Listening for H.264/HEVC packets on {server_address}")
    
    # Create a codec context for H.264/HEVC decoding
    codec_name = 'h264'  # Use 'hevc' for H.265
    codec = av.CodecContext.create(codec_name, 'r')
    
    # Buffer to store partial frames
    buffer = bytearray()
    
    # Initialize variable for FPS calculation
    prev_time = time.time()
    
    try:
        while True:
            try:
                # Receive the data packet
                data, address = sock.recvfrom(65536)  # Buffer size - adjust based on your needs
                
                # Add received data to buffer
                buffer.extend(data)
                
                # Try to decode frames from the buffer
                packet = av.Packet(buffer)
                frames = codec.decode(packet)
                
                # Display frames if any were decoded
                for frame in frames:
                    # Convert to numpy array for OpenCV
                    img = frame.to_ndarray(format='bgr24')
                    
                    # Calculate FPS for the current frame
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    
                    # Display FPS on the frame
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display the image
                    cv2.imshow('Frame from Raspberry Pi', img)
                    
                    # Clear buffer after successful decode
                    buffer = bytearray()
                    
                    # Press 'q' to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
            
            except socket.timeout:
                # Socket timeout - continue to next iteration
                continue
            
            except av.error.InvalidDataError:
                # If we can't decode the data yet, keep it in the buffer and continue
                # If buffer gets too large, we might need to reset it
                if len(buffer) > 1000000:  # 1MB limit
                    print("Buffer overflow, resetting")
                    buffer = bytearray()
                continue
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
