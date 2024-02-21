"""
AKA Sender
"""

import socket
import cv2
import sys
import argparse

def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")

def get_frames(rgb_path, depth_path):
    # Load the RGB frame as a color image
    rgb_frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_frame is None:
        raise ValueError(f"Failed to load RGB frame from {rgb_path}")

    # Load the Depth frame as is (without any color conversion)
    depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_frame is None:
        raise ValueError(f"Failed to load Depth frame from {depth_path}")

    return rgb_frame, depth_frame

def send_frame(client_socket, frame_type, frame):
    # Send frame type (e.g., 'RGB' or 'DEPTH')
    client_socket.sendall(frame_type)

    # Send the length of the frame data
    length = len(frame)
    client_socket.sendall(length.to_bytes(16, 'big'))  # Send the length as 16 bytes

    # Send the frame data
    client_socket.sendall(frame)

def send_frames(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    while not quit_keypress():
        rgb_frame, depth_frame = get_frames("./INFERENCE/rgb/sample/0001_0/00001.jpg", "./INFERENCE/depth/sample/0001_0/00001.jpg")
        
        # Encode frames before sending
        _, encoded_rgb_frame = cv2.imencode('.jpg', rgb_frame)
        _, encoded_depth_frame = cv2.imencode('.jpg', depth_frame)

        # Sending RGB Frame
        send_frame(client_socket, b'RGB', encoded_rgb_frame.tobytes())
        print("SENT RGB")

        # Sending Depth Frame
        send_frame(client_socket, b'DEPTH', encoded_depth_frame.tobytes())
        print("SENT DEPTH")

    # Sending the termination flag
    client_socket.sendall(b'STOP')

    # Receiving the inference result
    label = client_socket.recv(1024)
    print(f"Received label: {label.decode()}")

    client_socket.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Send frames over a network.')
    parser.add_argument('--receiver_ip', type=str, required=True, help='Receiver host IP')
    parser.add_argument('--port', type=int, required=True, help='Port number for communication.')
    args = parser.parse_args()
    port = args.port
    ip = args.receiver_ip

    HOST, PORT = ip, port
    window_name = "Press 0 to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 1024)
    background_image = cv2.imread('./images/background.png') 
    cv2.imshow(window_name, background_image)

    send_frames(HOST, PORT)
