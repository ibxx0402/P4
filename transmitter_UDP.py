import cv2
import socket
import math
import pickle
import sys
from picamera2 import Picamera2

max_length = 65000
host = "192.168.0.106"
port = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)}, lores={"size": (320, 240)}, display="main")
picam2.configure(config)

picam2.start()

#used to calculate the number of packets to be sent
frame = picam2.capture_array()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#print(frame.shape)
retval, buffer = cv2.imencode(".png", frame)

buffer = buffer.tobytes()
buffer_size = len(buffer)

num_of_packs = 1
if buffer_size > max_length:
    num_of_packs = math.ceil(buffer_size/max_length)

frame_info = {"packs":num_of_packs}
# send the number of packs to be expected
print("Number of packs:", num_of_packs)
sock.sendto(pickle.dumps(frame_info), (host, port))

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    retval, buffer = cv2.imencode(".jpg", frame)
    # convert to byte array
    buffer = buffer.tobytes()


    view = memoryview(buffer)
    for i in range(0, buffer_size, max_length):
        split_data = view[i:min(i+max_length, buffer_size)]
        sock.sendto(split_data, (host, port))

    # is only used for testing purposes
    #break
print("done")
