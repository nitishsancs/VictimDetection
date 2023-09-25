import cv2
from ultralytics import YOLO
import numpy as np
import torch

class_names = [
   "person", "skis", "snowboard",
    "cell phone"
]

cap = cv2.VideoCapture(0)
model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 230, 0), 3)
        class_name = class_names[cls]
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0, 230, 0), 3)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)  # Changed waitKey from 0 to 1 for smoother video playback

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()