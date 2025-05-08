import socket
import numpy as np
import cv2
import av
import time
import struct

def main():
    # Create a UDP socket and bind to port 9999 to receive forwarded video.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9995))
    sock.settimeout(2.0)

    server_address = ('130.225.164.144', 9999)
    # Send the registration message to the server.
    registration_message = b"Client registration"
    try:
        sock.sendto(registration_message, server_address)
        print(f"Client: Sent registration message to {server_address}")
        data, addr = sock.recvfrom(1024)  # Optionally wait for confirmation.
        print(f"Client: Received registration confirmation: {data}")
    except socket.timeout:
        print("Client: No registration confirmation received (continuing anyway)")

    print("Client: Listening for H.264 packets on port 9999")

    # Create a H.264 decoder using PyAV.
    codec = av.CodecContext.create('h264', 'r')

    # Buffer for non-fragmented data.
    buffer = bytearray()
    # Dictionary to store fragment reassemblies.
    fragments = {}

    prev_time = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue

        # Check if the packet is fragmented: our header is 10 bytes and starts with b'\xab\xcd'
        if len(data) >= 10 and data[:2] == b'\xab\xcd':
            header = data[:10]
            payload = data[10:]
            marker, frag_id, total_frags, frag_index = struct.unpack("!2sIHH", header)
            if frag_id not in fragments:
                fragments[frag_id] = {"total": total_frags, "chunks": {}}
            fragments[frag_id]["chunks"][frag_index] = payload

            # Reassemble if all fragments for this packet have arrived.
            if len(fragments[frag_id]["chunks"]) == total_frags:
                complete_data = bytearray()
                for i in range(1, total_frags + 1):
                    complete_data.extend(fragments[frag_id]["chunks"][i])
                del fragments[frag_id]
                buffer.extend(complete_data)
            else:
                # Wait for the remaining fragments.
                continue
        else:
            buffer.extend(data)

        # Attempt to decode a complete H.264 packet.
        try:
            packet = av.Packet(buffer)
            frames = codec.decode(packet)
            for frame in frames:
                img = frame.to_ndarray(format='bgr24')
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.imshow('Video Feed', img)
                buffer = bytearray()  # Reset buffer after a successful decode.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        except av.error.InvalidDataError:
            # If decoding fails (likely due to an incomplete packet), continue appending data.
            if len(buffer) > 1_000_000:
                print("Client: Buffer overflow, resetting")
                buffer = bytearray()
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nClient: Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
