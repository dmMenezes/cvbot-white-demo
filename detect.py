
from ultralytics import YOLO

# Load the model once globally
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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
