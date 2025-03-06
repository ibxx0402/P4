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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Serialize frame
    data = pickle.dumps(frame)

    frame_size = 4096
    end_index = 0
    data_length = len(data) 
    nr_split_frames = data_length // frame_size
    #print(nr_split_frames)
    leftover_frames = data_length % frame_size

    if leftover_frames !=0:
        nr_split_frames += 1

    #4 byte header including length of message, 3 byte for data_length
    header_1 = b'L' + data_length.to_bytes(3, 'big')
    s.sendto(header_1, address)

    #2 byte header including nr of messages 
    header_2 = b'S' + nr_split_frames.to_bytes(1, 'big')
    s.sendto(header_2, address)

    #ack that matches header files 
    while True:
        if s.recv(6) == header_1 + header_2:

            for i in range(nr_split_frames):
                start_index = end_index

                if (i+1) * frame_size > data_length:
                    end_index += leftover_frames
                
                else:
                    end_index += frame_size
                
                split_data = data[start_index : end_index]
                s.sendto(split_data, (host_ip, port))
            break
        else: 
            print("exit")
            exit()
