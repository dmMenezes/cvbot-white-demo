import asyncio
import numpy as np
from cvbot.communication.txtapiclient import TxtApiClient
from cvbot.controller.easy_drive_controller import EasyDriveController
from cvbot.config.drive_robot_configuration import DriveRobotConfiguration

from dotenv import load_dotenv
import os
import cv2
from ultralytics import YOLO

import warnings
import pathlib
import time
import threading

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
    print("Loading model...")
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize camera
print("Connecting to camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera!")
    exit()
else:
    print("Camera opened successfully")

# Set camera resolution lower to speed up (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Lock and update latest frame
            with self.lock:
                self.frame = frame
            # Flush out any other frames in buffer to keep latest only
            while self.cap.grab():
                ret2, frame2 = self.cap.retrieve()
                if ret2:
                    with self.lock:
                        self.frame = frame2

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()


def detect_ball_center(frame):
    results = model(frame)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf
        max_conf_idx = confidences.argmax().item()
        best_box = boxes.xyxy[max_conf_idx]
        x1, y1, x2, y2 = map(int, best_box.tolist())
        return True, (x1, y1, x2, y2)

    return False, (0, 0, 0, 0)


async def listen_for_quit(stop_event):
    while not stop_event.is_set():
        key = await asyncio.to_thread(input, "Press 'q' to quit: ")
        if key.lower() == 'q':
            stop_event.set()


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

    frame_grabber = FrameGrabber(cap)

    # PID controller params for left/right control
    Kp = 100.0
    Ki = 0.0
    Kd = 20.0
    integral = 0.0
    last_error = 0.0
    dt = 0.1  # approx loop interval in seconds

    forward_speed = 40.0  # forward speed value, adjust as needed

    while not stop_event.is_set():
        frame = frame_grabber.read()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        detected, box = detect_ball_center(frame)

        if detected:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            frame_height, frame_width = frame.shape[:2]
            bbox_height = y2 - y1

            error = (center_x / frame_width) - 0.5  # error from center line (-0.5 to 0.5)

            # PID calculations
            integral += error * dt
            derivative = (error - last_error) / dt
            output = Kp * error + Ki * integral + Kd * derivative
            last_error = error

            # Clamp output to speed limits (e.g., -100 to 100)
            max_speed = 100.0
            output = max(min(output, max_speed), -max_speed)

            # Decide forward speed based on bbox height
            # Move forward if bbox height < 90% frame height
            if bbox_height < 0.9 * frame_height:
                forward = forward_speed
            else:
                forward = 0.0  # stop forward movement if close enough

            print(f"Error: {error:.3f}, PID output: {output:.1f}, Forward speed: {forward:.1f}")

            # Draw bbox and center for display
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (int(center_x), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)
            text = f"Error: {error:.3f}, PID: {output:.1f}, Fwd: {forward:.1f}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save annotated frame
            filename = detected_dir / f"detected_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), annotated_frame)

            try:
                if forward_speed == 0.0:
                    # Command robot: speeds = [forward/backward, left/right, rotation]
                    await controller.drive(speeds=np.array([forward, output, 0.0]))
                else:
                    await controller.drive(speeds=np.array([0.0, 0.0, -100.0]))
            except Exception as e:
                print(f"Error during drive control: {e}")
                await controller.stop()
                continue

        else:
            # No detection: show frame and search by slowly moving left
            try:
                await controller.drive(speeds=np.array([0.0, 50.0, 0.0]))
                await asyncio.sleep(0.5)
                await controller.stop()
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error during search drive: {e}")
                continue

        await asyncio.sleep(dt)

    stop_event.set()
    quit_task.cancel()
    frame_grabber.stop()
    await controller.stop()
    cap.release()
    cv2.destroyAllWindows()


asyncio.run(connect())
