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
    sock.settimeout(50)
    header_1 = sock.recv(4)

    if header_1[:1] == b'L':
        sock.sendto(header_1, (client_ip, port))
        data = b''
        
        msg_size = int.from_bytes(header_1[1:], 'big')
        
        try:
            while len(data) < msg_size:
                sock.settimeout(0.1)
                data += sock.recv(1450)
            
            frame = pickle.loads(data)
            cv2.imshow('Video Stream', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        except Exception as e:
            continue
