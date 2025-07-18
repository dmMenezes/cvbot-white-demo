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

    # Grab a sample frame to get dimensions for video writer
    ret, sample_frame = cap.read()
    if not ret:
        print("Failed to grab sample frame for video writer")
        exit()
    frame_height, frame_width = sample_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = detected_dir / "output.avi"
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (frame_width, frame_height))

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

    # PID controller params for left/right control
    Kp = 100.0
    Ki = 0.0
    Kd = 20.0
    integral = 0.0
    last_error = 0.0
    dt = 0.1  # approx loop interval in seconds

    forward_speed = -100.0  # fixed forward speed (negative for forward)
    backward_speed = 100.0  # fixed backward speed (positive for backward)

    forward_motion = False
    backward_motion = False

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detected, box = detect_ball_center(frame)

        if detected:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            bbox_height = y2 - y1

            error = (center_x / frame_width) - 0.5  # error from center line (-0.5 to 0.5)

            # Annotate frame
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (int(center_x), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)

            text = f"Error: {error:.3f}, BBox H: {bbox_height}, Fwd:{forward_motion}, Bwd:{backward_motion}"
            cv2.putText(annotated_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write annotated frame to video
            out.write(annotated_frame)

            # Forward/backward motion logic

            # Check horizontal centering ranges
            centered_threshold = 0.05     # ±5% error range for forward motion
            outer_threshold = 0.07        # ±7% error range to stop forward/backward

            if forward_motion or backward_motion:
                # If centroid moves outside ±7%, stop forward/backward and go back to PID centering
                if abs(error) > outer_threshold:
                    forward_motion = False
                    backward_motion = False
                    await controller.stop()
                    # Reset PID integral and derivative to avoid spikes
                    integral = 0.0
                    last_error = error
                    print("Centroid out of range during forward/backward motion, switching to PID centering")
                else:
                    # Continue forward/backward motion based on bbox height
                    if bbox_height < 0.8 * frame_height:
                        # Move forward
                        await controller.drive(speeds=np.array([0.0, 0.0, forward_speed]))
                        print(f"Moving forward: bbox_height={bbox_height}")
                    elif bbox_height > 0.9 * frame_height:
                        # Move backward
                        await controller.drive(speeds=np.array([0.0, 0.0, backward_speed]))
                        print(f"Moving backward: bbox_height={bbox_height}")
                    else:
                        # Stop motion if bbox height in acceptable range
                        forward_motion = False
                        backward_motion = False
                        await controller.stop()
                        print("Stopped forward/backward motion: bbox height acceptable")
            else:
                # Use PID to center horizontally if not in forward/backward motion
                if abs(error) > centered_threshold:
                    # PID calculations
                    integral += error * dt
                    derivative = (error - last_error) / dt
                    output = Kp * error + Ki * integral + Kd * derivative
                    last_error = error

                    # Clamp output
                    max_speed = 100.0
                    output = max(min(output, max_speed), -max_speed)

                    try:
                        await controller.drive(speeds=np.array([0.0, output, 0.0]))
                        print(f"PID centering: error={error:.3f}, output={output:.1f}")
                    except Exception as e:
                        print(f"Error during drive control: {e}")
                        await controller.stop()
                        continue
                else:
                    # Centroid within ±5%, start forward motion if bbox height < 80%
                    if bbox_height < 0.8 * frame_height:
                        forward_motion = True
                        backward_motion = False
                        await controller.drive(speeds=np.array([0.0, 0.0, forward_speed]))
                        print("Starting forward motion")
                    else:
                        await controller.stop()
                        print("Centered and close enough, stopped")

        else:
            # No detection: write raw frame and search by slowly moving left
            out.write(frame)

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
    await controller.stop()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

asyncio.run(connect())
