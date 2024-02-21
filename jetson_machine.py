import socket
import os
import argparse
import cv2
import numpy as np
import time

def inference():
    # Implement the inference logic here
    # Placeholder return
    print("Doing inference.....")
    return 0

def save_frame(frame, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Assuming frame is a numpy array
    filename = os.path.join(folder, f"frame_{int(time.time())}.jpg")
    cv2.imwrite(filename, frame)

def receive_and_assemble_frame(conn):
    try:
        # Receive the length of the frame
        length_bytes = conn.recv(16)
        length = int.from_bytes(length_bytes, 'big')  # Interpret the bytes as an integer

        # Receive the actual frame
        frame_data = b''
        while len(frame_data) < length:
            packet = conn.recv(min(length - len(frame_data), 4096))
            if not packet: break
            frame_data += packet

        # Decode the frame
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode frame")

        return frame
    except Exception as e:
        print(f"Error in receive_and_assemble_frame: {e}")
        return None


def receive_frames(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    conn, addr = server_socket.accept()

    while True:
        data = conn.recv(1024)
        if not data:
            break

        if data == b'RGB':
            rgb_frame = receive_and_assemble_frame(conn)
            if rgb_frame is not None:
                print("Saving RGB")
                save_frame(rgb_frame, 'rgb_frames')

        elif data == b'DEPTH':
            depth_frame = receive_and_assemble_frame(conn)
            if depth_frame is not None:
                print("Saving DEPTH")
                save_frame(depth_frame, 'depth_frames')

        elif data == b'STOP':
            label = inference()
            conn.sendall(str(label).encode())
            break

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Receive frames over a network.')
    parser.add_argument('--port', type=int, required=True, help='List of port numbers for communication with cameras.')
    args = parser.parse_args()
    receiver_port = args.port

    HOST, PORT = "0.0.0.0", receiver_port  # Bind to all interfaces
    receive_frames(HOST, PORT)
