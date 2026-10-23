import cv2

from yolo import YOLOSystem


print("Starting Arachne YOLO module...")

yolo = YOLOSystem()

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()

print("Camera: ONLINE")
print("YOLO module: READY")


while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")

        break

    yolo.run(frame)

    detections = yolo.get_detections()

    display_frame = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        class_name = detection["class_name"]

        confidence = detection["confidence"]

        object_id = detection["object_id"]
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"ID {object_id} | {class_name} {confidence:.2f}"

        cv2.putText(
            display_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "ARACHNE YOLO MODULE TEST",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()

print("YOLO MODULE OFFLINE")