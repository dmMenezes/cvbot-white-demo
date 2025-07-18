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
            frame_height, frame_width = frame.shape[:2]
            bbox_height = y2 - y1

            error = (center_x / frame_width) - 0.5  # error from center line (-0.5 to 0.5)

            # Draw bbox and center for display
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (int(center_x), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)
            text = f"Error: {error:.3f}, BBox height: {bbox_height}, Forward: {forward_motion}, Backward: {backward_motion}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save annotated frame
            filename = detected_dir / f"detected_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), annotated_frame)

            if forward_motion or backward_motion:
                # If moving forward/backward, check if centroid drifts beyond ±7%
                if abs(error) > 0.07:
                    # Stop forward/backward motion and switch to centering
                    forward_motion = False
                    backward_motion = False
                    await controller.stop()
                    print("Centroid drifted out of range, stopping forward/backward motion.")
                else:
                    # Continue forward or backward motion based on bbox height
                    if forward_motion and bbox_height >= 0.9 * frame_height:
                        # Too close, start moving backward
                        forward_motion = False
                        backward_motion = True
                        print("Too close, switching to backward motion.")
                    elif backward_motion and bbox_height <= 0.8 * frame_height:
                        # Far enough, stop backward motion
                        backward_motion = False
                        await controller.stop()
                        print("Back to acceptable distance, stopping backward motion.")

                    try:
                        if forward_motion:
                            # Move forward
                            await controller.drive(speeds=np.array([0.0, 0.0, -100.0]))
                            print("Moving forward.")
                        elif backward_motion:
                            # Move backward
                            await controller.drive(speeds=np.array([0.0, 0.0, 100.0]))
                            print("Moving backward.")
                        else:
                            # Stop motion
                            await controller.stop()
                    except Exception as e:
                        print(f"Error during forward/backward drive: {e}")
                        await controller.stop()
            else:
                # Not moving forward/backward, do PID centering until within ±5%
                if abs(error) <= 0.05:
                    # Within acceptable range, start forward motion if bbox height < 80%
                    if bbox_height < 0.8 * frame_height:
                        forward_motion = True
                        print("Starting forward motion.")
                        try:
                            await controller.drive(speeds=np.array([0.0, 0.0, -100.0]))
                        except Exception as e:
                            print(f"Error starting forward drive: {e}")
                            await controller.stop()
                    else:
                        # Close enough, no forward motion needed
                        await controller.stop()
                else:
                    # PID left/right centering
                    integral += error * dt
                    derivative = (error - last_error) / dt
                    output = Kp * error + Ki * integral + Kd * derivative
                    last_error = error

                    # Clamp output
                    max_speed = 100.0
                    output = max(min(output, max_speed), -max_speed)

                    try:
                        await controller.drive(speeds=np.array([0.0, output, 0.0]))
                        print(f"Centering with PID output: {output:.1f}")
                    except Exception as e:
                        print(f"Error during PID centering drive: {e}")
                        await controller.stop()

            # Show live video
            cv2.imshow('Detection', annotated_frame)

        else:
            # No detection: show frame and search by slowly moving left
            cv2.imshow('Detection', frame)
            try:
                await controller.drive(speeds=np.array([0.0, 50.0, 0.0]))
                await asyncio.sleep(0.5)
                await controller.stop()
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error during search drive: {e}")
                continue

        # Allow quit on window keypress 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        await asyncio.sleep(dt)

    stop_event.set()
    quit_task.cancel()
    await controller.stop()
    cap.release()
    cv2.destroyAllWindows()

asyncio.run(connect())
