import cv2
from ultralytics import YOLO

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO: ONLINE")
print("Starting camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()

print("Camera: ONLINE")

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")
        break

    results = model(
        frame,
        imgsz=320,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "ARACHNE YOLO TEST",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

camera.release()

cv2.destroyAllWindows()

print("YOLO TEST OFFLINE")