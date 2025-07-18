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

# Suppress specific Pydantic warning
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main"
)


load_dotenv()  # loads .env from current dir

HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
KEY = os.getenv("KEY")
# print(HOST, PORT, KEY)
# print(type(HOST), type(PORT), type(KEY))

# Load the model once globally
try:
    print("loading model")
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("connecting to camera")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera!")
    exit()
else:
    print("Camera opened successfully")


def detect_ball_center(frame):
    """
    Detect the most confident ball in the given frame.

    Returns:
        (bool, tuple): 
            - True and (x1, y1, x2, y2) if ball detected
            - False and (0, 0, 0, 0) if no detection
    """
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
    # The following classes are needed to init the drive controller
    try:
        print("Initializing API client and controller")
         # Initialize the API client and controller
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
            # await api_client.close()
            break

        # print("true")
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

                elif norm_y > 0.3 and norm_x < 0.55 and norm_x > 0.45:
                    await controller.drive(speeds=np.array([0.0, 0.0, -100.0]))

                else:
                    await controller.stop()
                

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw center dot
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                # Show normalized coords on image
                text = f"X: {norm_x:.3f}, Y: {norm_y:.3f}"
                cv2.putText(annotated_frame, text, (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # print(f"Ball detected at normalized coords: X={norm_x:.3f}, Y={norm_y:.3f}")
            
            except Exception as e:
                print(f"Error during drive control: {e}")
                await controller.stop()
                continue
        
        if not detected:
            try:
                # print("No ball detected, trying to find...")
                # await api_client.initialize()
                await controller.drive(speeds=np.array([0.0, 100.0, 0.0]))
            except Exception as e:
                print(f"Error during stop: {e}")
                continue


        headless = not os.environ.get("DISPLAY")

        # Later in code, only show frame if GUI is available
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

asyncio.run(connect())