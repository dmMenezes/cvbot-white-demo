import asyncio
import matplotlib.pyplot as plt
import numpy as np

from cvbot.communication import controller
from cvbot.communication.txtapiclient import TxtApiClient
from cvbot.controller.easy_drive_controller import EasyDriveController
from cvbot.config.drive_robot_configuration import DriveRobotConfiguration

from dotenv import load_dotenv
import os
import cv2
from ultralytics import YOLO

import warnings
import signal

# Suppress specific Pydantic warning
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main"
)

load_dotenv()

HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
KEY = os.getenv("KEY")

# Load the YOLO model
try:
    print("loading model")
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize camera
print("connecting to camera")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera!")
    exit()
else:
    print("Camera opened successfully")

def detect_ball_center(frame):
    results = model(frame)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf
        max_conf_idx = confidences.argmax().item()
        best_box = boxes.xyxy[max_conf_idx]
        x1, y1, x2, y2 = best_box.tolist()
        return True, (x1, y1, x2, y2)

    return False, (0, 0, 0, 0)

async def listen_for_quit(stop_event):
    while not stop_event.is_set():
        key = await asyncio.to_thread(input, "Press 'q' to quit: ")
        if key.lower() == 'q':
            stop_event.set()

import time

# Make sure to create this directory at the start of connect()
import pathlib

async def connect():
    detected_dir = pathlib.Path("detected")
    detected_dir.mkdir(exist_ok=True)

    stop_event = asyncio.Event()
    quit_task = asyncio.create_task(listen_for_quit(stop_event))

    try:
        print("Initializing API client and controller")
        api_client = TxtApiClient(HOST, PORT, KEY)
        controller = EasyDriveController(api_client, DriveRobotConfiguration())
        await api_client.initialize()
        await controller.stop()

    except Exception as e:
        print(f"Error initializing API client or controller: {e}")
        exit()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab the frame")
            break

        detected, box = detect_ball_center(frame)

        if detected:
            x1, y1, x2, y2 = map(int, box)
            # Draw bounding box and center
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            text = f"Center: ({center_x},{center_y})"
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save the annotated frame with timestamp
            filename = detected_dir / f"detected_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), annotated_frame)

            await controller.stop()
            await asyncio.sleep(5)
            frame_height, frame_width = frame.shape[:2]
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            bb_height = y2 - y1

            try:
                if bb_height > 0.9 * frame_height:
                    await controller.stop()
                elif norm_x <= 0.30:
                    await controller.drive(speeds=np.array([0.0, 30.0, 0.0]))
                elif norm_x >= 0.70:
                    await controller.drive(speeds=np.array([0.0, -30.0, 0.0]))
                elif norm_y > 0.2 and 0.30 < norm_x < 0.70:
                    await controller.drive(speeds=np.array([0.0, 0.0, -30.0]))
                else:
                    await controller.stop()
            except Exception as e:
                print(f"Error during drive control: {e}")
                await controller.stop()
                continue

        else:
            try:
                await controller.drive(speeds=np.array([0.0, 30.0, 0.0]))
            except Exception as e:
                print(f"Error during drive attempt: {e}")
                continue

    stop_event.set()
    quit_task.cancel()
    await controller.stop()
    cap.release()
    cv2.destroyAllWindows()



asyncio.run(connect())
