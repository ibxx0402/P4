import socket
import select
import struct

# Server configuration
SERVER_IP = '192.168.0.119'
CLIENT_PORT = 9999  # Port for client registration and for sending video out
PI_PORT = 9998      # Port for receiving the Raspberry Pi video feed

# Maximum UDP payload per datagram (safe size)
MAX_CHUNK_SIZE = 1400

def main():
    client_address = None
    frag_counter = 0

    # Create a UDP socket for client registration/messages (port 9999)
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.bind((SERVER_IP, CLIENT_PORT))

    # Create a UDP socket for receiving Raspberry Pi video feed (port 9998)
    pi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_sock.bind((SERVER_IP, PI_PORT))

    print(f"Server: Listening for client registration on {SERVER_IP}:{CLIENT_PORT}")
    print(f"Server: Listening for Pi video feed on {SERVER_IP}:{PI_PORT}")

    # Use select to wait for activity on either socket
    sockets = [client_sock, pi_sock]

    while True:
        ready, _, _ = select.select(sockets, [], [])
        for sock in ready:
            if sock == client_sock:
                # Handle a registration or control message from the client.
                data, addr = client_sock.recvfrom(1024)
                print(f"Server: Received client message from {addr}: {data}")
                client_address = addr
                # Optionally, send a confirmation.
                client_sock.sendto(b'Registration successful', addr)
            elif sock == pi_sock:
                # Handle a video packet from the Raspberry Pi.
                data, addr = pi_sock.recvfrom(65536)
                if client_address is None:
                    print("Server: No client registered; dropping video packet.")
                    continue
                # Check the data size: if it’s too large, fragment it.
                if len(data) <= MAX_CHUNK_SIZE:
                    client_sock.sendto(data, client_address)
                else:
                    # Fragment the data
                    frag_id = frag_counter
                    frag_counter = (frag_counter + 1) % (2**32)
                    total_frags = (len(data) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
                    for i in range(total_frags):
                        chunk = data[i * MAX_CHUNK_SIZE:(i + 1) * MAX_CHUNK_SIZE]
                        # The header format: 2-byte marker, 4-byte frag_id, 2-byte total_frag, 2-byte current fragment index.
                        header = struct.pack("!2sIHH", b'\xab\xcd', frag_id, total_frags, i + 1)
                        client_sock.sendto(header + chunk, client_address)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer: Shutting down.")
