import threading
import time

from ultralytics import YOLO


class YOLOSystem:

    def __init__(self):

        print("Loading YOLO...")

        self.model = YOLO("yolo11n.pt")

        self.results = None
        self.detections = []

        self.enabled = True
        self.running = False

        self.lock = threading.Lock()

        # YOLO SETTINGS
        self.detection_interval = 0.5
        self.confidence_threshold = 0.25

        self.class_history = {}
        self.class_history_length = 5

        self.last_run_time = 0

        self.next_object_id = 1
        self.tracked_objects = {}
        
        # -------------------------
        # STALE OBJECT CLEANUP
        # -------------------------

        self.object_timeout = 2.0

        print("YOLO: ONLINE")

    def run(self, frame):

        if not self.enabled:
            return

        current_time = time.time()

        if current_time - self.last_run_time < self.detection_interval:
            return

        if self.running:
            return

        self.running = True
        self.last_run_time = current_time

        thread = threading.Thread(
            target=self._inference,
            args=(frame.copy(),),
            daemon=True
        )

        thread.start()

  # Update history of detected classes
    def _update_class_history(self, class_id):

        if class_id not in self.class_history:

            self.class_history[class_id] = []

        history = self.class_history[class_id]

        history.append(class_id)

        if len(history) > self.class_history_length:

            history.pop(0)

        return history
    
    def _calculate_center(self, box):

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
            
        return center_x, center_y


    # Find the closest tracked object to the given bounding box
    def _find_matching_object(self, box, class_id):

       center_x, center_y = self._calculate_center(box)

       closest_id = None
       closest_distance = float("inf")

       for object_id, object_data in self.tracked_objects.items():

           previous_class_id = object_data["class_id"]

           if previous_class_id != class_id:

               continue

           previous_x = object_data["center"][0]
           previous_y = object_data["center"][1]

           distance = (
               (center_x - previous_x) ** 2
               +
               (center_y - previous_y) ** 2
            ) ** 0.5

           if distance < closest_distance:

               closest_distance = distance
               closest_id = object_id

       if closest_distance < 200:

           return closest_id

       return None
    
    def _inference(self, frame):

        try:

            results = self.model(
                frame,
                imgsz=640,
                conf=self.confidence_threshold,
                verbose=False
            )

            detections = []

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    coordinates = box.xyxy[0].tolist()

                    x1, y1, x2, y2 = coordinates

                    box = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )

                    object_id = self._find_matching_object(
                        box,
                        class_id
                    )

                    if object_id is None:

                        object_id = self.next_object_id

                        self.next_object_id += 1

                    center = self._calculate_center(box)

                                        # -------------------------
                    # CLASSIFICATION HISTORY
                    # -------------------------

                    if object_id not in self.class_history:

                        self.class_history[object_id] = []

                    history = self.class_history[object_id]

                    history.append(class_id)

                    if len(history) > self.class_history_length:

                        history.pop(0)

                    class_counts = {}

                    for historical_class in history:

                        if historical_class not in class_counts:

                            class_counts[historical_class] = 0

                        class_counts[historical_class] += 1

                    stable_class_id = max(
                        class_counts,
                        key=class_counts.get
                    )

                    stable_class_name = self.model.names[
                        stable_class_id
                    ]

                    self.tracked_objects[object_id] = {
                        "center": center,
                        "class_id": stable_class_id,
                        "class_name": stable_class_name,
                        "confidence": confidence,
                        "box": box,
                        "last_seen": time.time()
                    }
                    
                    detections.append({

                        "object_id": object_id,

                        "class_id": stable_class_id,

                        "class_name": stable_class_name,

                        "confidence": confidence,

                        "box": box

                    })
                    # -------------------------
                    # REMOVE STALE OBJECTS
                    # -------------------------

                    current_time = time.time()

                    stale_objects = []

                    for object_id, object_data in self.tracked_objects.items():

                        time_since_seen = (
                            current_time
                            -
                            object_data["last_seen"]
                        )

                        if time_since_seen > self.object_timeout:

                            stale_objects.append(object_id)


                    for object_id in stale_objects:

                        del self.tracked_objects[object_id]

                        if object_id in self.class_history:

                            del self.class_history[object_id]

                            
            with self.lock:

                self.results = results
                self.detections = detections

        except Exception as error:

            print("YOLO ERROR:", error)

        finally:

            self.running = False

    def enable(self):

        self.enabled = True

        print("YOLO FEATURE: ENABLED")

    def disable(self):

        self.enabled = False

        print("YOLO FEATURE: DISABLED")

    def get_results(self):

        with self.lock:

            return self.results

    def get_detections(self):

        with self.lock:

            return self.detections.copy()