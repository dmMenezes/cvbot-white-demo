import cv2
from ultralytics import YOLO

# Load the model once globally
model = YOLO('best.pt')

def detect_ball_center(frame):
    """
    Detect the most confident ball in the given frame.

    Returns:
        (bool, tuple): 
            - True and (x1, y1, x2, y2) if ball detected
            - False and (0, 0, 0, 0) if no detection
    """
    results = model(frame)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf
        max_conf_idx = confidences.argmax().item()

        best_box = boxes.xyxy[max_conf_idx]
        x1, y1, x2, y2 = best_box.tolist()

        return True, (x1, y1, x2, y2)

    return False, (0, 0, 0, 0)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera!")
    exit()

while True:
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

        # Draw bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw center dot
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        # Show normalized coords on image
        text = f"X: {norm_x:.3f}, Y: {norm_y:.3f}"
        cv2.putText(annotated_frame, text, (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"Ball detected at normalized coords: X={norm_x:.3f}, Y={norm_y:.3f}")

    cv2.imshow("YOLOv8 Football Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
