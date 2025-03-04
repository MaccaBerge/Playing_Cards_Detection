import cv2
from sys import exit
from ultralytics import YOLO

model = YOLO("runs/detect/yolov8n_custom/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not open webcam.")
        break

    detect_result = model(frame)
    detect_image = detect_result[0].plot()

    cv2.imshow("Playing Card Detection", detect_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

