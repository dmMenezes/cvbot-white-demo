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

def detect_ball_center(frame, verbose=False):
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
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            frame_height, frame_width = frame.shape[:2]
            LR_center = frame_width / 2
            print(f"Center X: {center_x}, LR_center: {LR_center}")
            # x1, y1, x2, y2 = map(int, box)
            # center_x = int((x1 + x2) / 2)
            # center_y = int((y1 + y2) / 2)
            # frame_height, frame_width = frame.shape[:2]
            # norm_x = (center_x / frame_width) - 0.5  # normalized x centered at 0
            # print(f"Normalized X: {norm_x:.3f}")

            # Draw bounding box and center on copy of frame
            # annotated_frame = frame.copy()
            # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            # text = f"Center X: {norm_x:.3f}"
            # cv2.putText(annotated_frame, text, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save annotated frame with timestamp
            # filename = detected_dir / f"detected_{int(time.time() * 1000)}.jpg"
            # cv2.imwrite(str(filename), annotated_frame)

        #     try:
        #         await controller.stop()  # Ensure stopped before moving
        #         await asyncio.sleep(1)

        #         center_threshold = 0.1

        #         if norm_x < -center_threshold:
        #             # Distance from deadzone edge on left side
        #             diff = abs(norm_x + center_threshold)
        #             # Clamp diff between 0.0 and 0.35 (because max diff = 0.45 - 0.1)
        #             diff = max(0.0, min(diff, 0.35))
        #             # Scale sleep time from 0.1 (min) to 0.5 (max)
        #             sleep_time = 0.05 + (diff / 0.35) * (0.5 - 0.1)

        #             await controller.drive(speeds=np.array([0.0, 50.0, 0.0]))
        #             await asyncio.sleep(sleep_time)
        #             await controller.stop()
        #             await asyncio.sleep(0.5)
        #             print(f"Moved left for {sleep_time:.2f}s")

        #         elif norm_x > center_threshold:
        #             # Distance from deadzone edge on right side
        #             diff = abs(norm_x - center_threshold)
        #             diff = max(0.0, min(diff, 0.35))
        #             sleep_time = 0.05 + (diff / 0.35) * (0.5 - 0.1)

        #             await controller.drive(speeds=np.array([0.0, -50.0, 0.0]))
        #             await asyncio.sleep(sleep_time)
        #             await controller.stop()
        #             await asyncio.sleep(0.5)
        #             print(f"Moved right for {sleep_time:.2f}s")

        #         else:
        #             # Inside deadzone, no movement
        #             await controller.stop()
        #             await asyncio.sleep(0.5)

        #     except Exception as e:
        #         print(f"Error during drive control: {e}")
        #         await controller.stop()
        #         continue

        # else:
        #     try:
        #         # Move left for 0.5s, then stop and wait 0.5s
        #         await controller.drive(speeds=np.array([0.0, 50.0, 0.0]))
        #         await asyncio.sleep(0.2)
        #         await controller.stop()
        #         await asyncio.sleep(0.5)
        #     except Exception as e:
        #         print(f"Error during drive attempt: {e}")
        #         continue

    stop_event.set()
    quit_task.cancel()
    await controller.stop()
    cap.release()
    cv2.destroyAllWindows()




asyncio.run(connect())
