"""
AKA Sender
"""
import struct
import pickle
import zlib
import socket
import argparse
import numpy as np
import sys
import cv2
import time
import os
import subprocess
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device_ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument('--receiver_ip', type=str, required=True, help='Receiver host IP')
    parser.add_argument('--port', type=int, required=True, help='Port number for communication.')

    return parser.parse_args()

class StreamingClientObserver():
    def __init__(self):
        self.images = {}

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")

def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)

def device_stream(args):
    if sys.platform.startswith("linux"):
        update_iptables()
    # Set debug level
    aria.set_log_level(aria.Level.Info)
    # Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    # Connect to the device
    device = device_client.connect()
    # Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client
    # Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    # Streaming type
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    # Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    # Start streaming
    streaming_manager.start_streaming()
    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")
    return streaming_manager, streaming_client, device_client, device

def device_subscribe(streaming_client):
    # Configure subscription
    config = streaming_client.subscription_config
    config.subscriber_data_type = (aria.StreamingDataType.Rgb)
    # Take most recent frame
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    # Set the observer
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    # Start listening
    print("Start listening to image data")
    streaming_client.subscribe()
    return observer

def send_frame(client_socket, frame_type, frame):
    # Send frame type (e.g., 'RGB' or 'DEPTH')
    client_socket.sendall(frame_type)

    # Send the length of the frame data
    length = len(frame)
    client_socket.sendall(length.to_bytes(16, 'big'))  # Send the length as 16 bytes

    # Send the frame data
    client_socket.sendall(frame)

def send_frames(host, port, frame):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    while not quit_keypress():
        rgb_frame = frame
        
        # Encode frames before sending
        _, encoded_rgb_frame = cv2.imencode('.jpg', rgb_frame)

        # Sending RGB Frame
        send_frame(client_socket, b'RGB', encoded_rgb_frame.tobytes())
        print("SENT RGB")

    # Sending the termination flag
    client_socket.sendall(b'STOP')

    # Receiving the inference result
    label = client_socket.recv(1024)
    print(f"Received label: {label.decode()}")

    client_socket.close()


    # Main function
def main():
    args = parse_args()
    streaming_manager, streaming_client, device_client, device = device_stream(args)
    observer = device_subscribe(streaming_client)

    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.moveWindow(rgb_window, 50, 50)
    
    HOST, PORT = args.ip, args.port

    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket.connect((HOST, PORT))

    while not quit_keypress():
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

            # Serialize and compress the frame
            serialized_frame = pickle.dumps(rgb_image)
            compressed_frame = zlib.compress(serialized_frame)

            # Send the size of the compressed frame first
            sender_socket.sendall(struct.pack(">L", len(compressed_frame)))

            # Send the compressed frame
            sender_socket.sendall(compressed_frame)

    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    sender_socket.close()
