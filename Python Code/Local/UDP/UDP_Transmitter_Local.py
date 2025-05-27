import cv2
import socket
import math
import threading
import queue
from picamera2 import Picamera2

max_length = 65000
host = "192.168.0.100"
port = 5000

im_enc = "."+"jpg"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(0)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"}, # directly capture in RGB format
    raw={'size': (2304, 1296)},
    buffer_count=10,
    #controls={'FrameRate': 15},
)
picam2.configure(config)

picam2.start()

# Create a queue for frames
frame_queue = queue.Queue(maxsize=10)

# Thread for capturing frames
def capture_thread():
    while True:
        frame = picam2.capture_array()
        retval, buffer = cv2.imencode(im_enc, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not frame_queue.full() and retval:
            frame_queue.put(buffer.tobytes())

# Start the capture thread
threading.Thread(target=capture_thread, daemon=True).start()

# Main thread handles network transmission
while True:
    try:
        buffer = frame_queue.get(timeout=1.0)
        buffer_size = len(buffer)
        num_of_packs = math.ceil(buffer_size/max_length)
        header = num_of_packs.to_bytes(3, 'big')
        sock.sendto(header, (host,port))
        
        view = memoryview(buffer)
        for i in range(0, buffer_size, max_length):
            split_data = view[i:min(i+max_length, buffer_size)]
            sock.sendto(split_data, (host, port))
    except queue.Empty:
        continue

