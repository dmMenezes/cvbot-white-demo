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
import threading
from flask import Flask, Response

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

# Shared annotated frame and lock for streaming
current_annotated_frame = None
frame_lock = threading.Lock()

# Flask app for video streaming
app = Flask(__name__)

def generate_stream():
    global current_annotated_frame
    while True:
        with frame_lock:
            if current_annotated_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', current_annotated_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

def detect_ball_center(frame):
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf
        max_conf_idx = confidences.argmax().item()
        best_box = boxes.xyxy[max_conf_idx]
        x1, y1, x2, y2 = best_box.tolist()
        return True, (x1, y1, x2, y2)

    return False, (0, 0, 0, 0)

async def connect():
    try:
        print("Initializing API client and controller")
        api_client = TxtApiClient(HOST, PORT, KEY)
        controller = EasyDriveController(api_client, DriveRobotConfiguration())
        await api_client.initialize()
        await controller.stop()
    
    except Exception as e:
        print(f"Error initializing API client or controller: {e}")
        exit()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            await controller.stop()
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab the frame")
            break

        detected, box = detect_ball_center(frame)
        annotated_frame = frame.copy()

        if detected:
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            frame_height, frame_width = frame.shape[:2]
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            bb_height = y2 - y1

            try:
                if bb_height > 0.9 * frame_height:
                    await controller.stop()
                elif norm_x <= 0.45:
                    await controller.drive(speeds=np.array([0.0, 100.0, 0.0]))
                elif norm_x >= 0.55:
                    await controller.drive(speeds=np.array([0.0, -100.0, 0.0]))
                elif norm_y > 0.3 and 0.45 < norm_x < 0.55:
                    await controller.drive(speeds=np.array([0.0, 0.0, -100.0]))
                else:
                    await controller.stop()

                # Draw annotations
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                text = f"X: {norm_x:.3f}, Y: {norm_y:.3f}"
                cv2.putText(annotated_frame, text, (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error during drive control: {e}")
                await controller.stop()
                continue
        else:
            try:
                await controller.drive(speeds=np.array([0.0, 100.0, 0.0]))
            except Exception as e:
                print(f"Error during drive attempt: {e}")
                continue

        # Store frame for both GUI and HTTP
        with frame_lock:
            current_annotated_frame = annotated_frame.copy()

        # GUI display (preserved)
        headless = not os.environ.get("DISPLAY")
        if not headless:
            cv2.imshow("Football Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    await controller.stop()
    cap.release()
    cv2.destroyAllWindows()

    # first param turns front and rear in reverse
    # second param rotate positive -> left
    # third param forward (negative)
    # await asyncio.sleep(1.0)


async def main():
    stop_event = asyncio.Event()

    def shutdown():
        print("\n[INFO] Shutdown signal received. Stopping...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("[INFO] Flask streaming server started at http://<rpi-ip>:5000/video_feed")

    connect_task = asyncio.create_task(connect())
    await stop_event.wait()
    connect_task.cancel()
    try:
        await connect_task
    except asyncio.CancelledError:
        print("[INFO] connect() task cancelled cleanly.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Program interrupted manually.")
