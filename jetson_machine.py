import socket
import pickle
import zlib
import cv2
import numpy as np
import struct

def receive_frame(conn):
    # Read the size of the compressed frame
    data = b""
    while len(data) < 4:
        data += conn.recv(4 - len(data))
    frame_size = struct.unpack(">L", data)[0]

    # Receive the compressed frame
    frame_data = b""
    while len(frame_data) < frame_size:
        packet = conn.recv(min(frame_size - len(frame_data), 4096))
        if not packet:
            return None
        frame_data += packet

    # Decompress and deserialize the frame
    frame_data = zlib.decompress(frame_data)
    frame = pickle.loads(frame_data)
    return frame

def main():
    HOST, PORT = '0.0.0.0', 12345

    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver_socket.bind((HOST, PORT))
    receiver_socket.listen(1)
    conn, addr = receiver_socket.accept()

    while True:
        frame = receive_frame(conn)
        if frame is None:
            break

        cv2.imshow('Received Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    conn.close()
    receiver_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
