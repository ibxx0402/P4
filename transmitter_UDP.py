# Server Side (Raspberry Pi)
import cv2
import numpy as np
import socket
import pickle
from picamera2 import Picamera2

BUFF_SIZE = 65536
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_name = socket.gethostname()
host_ip = "192.168.0.111"
client_ip = "192.168.0.101"
port = 65432
address = (host_ip, port)
s.bind((client_ip, port))

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)}, lores={"size": (320, 240)}, display="main")
picam2.configure(config)


picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Optional: Process frame
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Serialize frame
    data = pickle.dumps(frame)

    #size chosen because of the MTU size of 1500 bytes
    frame_size = 1450
    data_length = len(data) 
    nr_split_frames = data_length // frame_size
    #print(nr_split_frames)
    leftover_frames = data_length % frame_size

    if leftover_frames !=0:
        nr_split_frames += 1

    #4 byte header including length of message, 3 byte for data_length
    header_1 = b'L' + data_length.to_bytes(3, 'big')
    s.sendto(header_1, address)

    bytes_sent = 0
    
    # Pre-allocate the view once and reuse
    data_view = memoryview(bytearray(frame_size))

    try:
        while True:
            s.settimeout(0.1)
            if s.recv(4) == header_1:

                view = memoryview(data)
                for i in range(0, data_length, frame_size):
                    split_data = view[i:min(i+frame_size, data_length)]
                    #print(len(split_data)) 
                    bytes_sent += s.sendto(split_data, (host_ip, port))
                break
             
            else:
                break
    except socket.timeout:
        continue
