import cv2
import numpy as np
import socket
import pickle


BUFF_SIZE = 65536
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_name = socket.gethostname()
host_ip = "192.168.0.111"
client_ip = "192.168.0.101"
port = 65432
socket_address = (host_ip, port)
sock.bind(socket_address)


while True:
    header_1 = sock.recv(4)
    header_2 = sock.recv(2)

    if header_1[:1] != b'L' or header_2[:1] != b'S':
        sock.sendto(b'F', (client_ip, port))
    else: 
        sock.sendto(header_1 + header_2, (client_ip, port))
        data = b''
        
        while len(data) < int.from_bytes(header_1[1:], 'big'):
            data += sock.recv(4096)
        
        frame = pickle.loads(data)
        data = b''
        cv2.imshow('Video Stream', frame)

        # Break loop if 'q' is pressed is necessary for window not closing
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
