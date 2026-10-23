# Imports
import cv2
import time
import threading

from yolo import YOLOSystem

# -------------------------
# YOLO TARGET SCORING
# -------------------------

def calculate_yolo_target_score(
    detection,
    tracked_position,
    current_yolo_target_id
):

    x1, y1, x2, y2 = detection["box"]

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    width = x2 - x1
    height = y2 - y1

    confidence = detection["confidence"]

    # -------------------------
    # CONFIDENCE SCORE
    # -------------------------

    confidence_score = confidence * 40

    # -------------------------
    # SIZE SCORE
    # -------------------------

    object_area = width * height

    size_score = min(
        object_area / 10000,
        1
    ) * 20

    # -------------------------
    # DISTANCE SCORE
    # -------------------------

    distance_score = 0

    if tracked_position is not None:

        distance = (
            (
                center_x
                -
                tracked_position[0]
            ) ** 2
            +
            (
                center_y
                -
                tracked_position[1]
            ) ** 2
        ) ** 0.5

        distance_score = max(
            0,
            30 - (distance / 10)
        )

    # -------------------------
    # CURRENT TARGET BONUS
    # -------------------------

    target_bonus = 0

    if (
        current_yolo_target_id is not None
        and
        detection["object_id"]
        ==
        current_yolo_target_id
    ):

        target_bonus = 20

    # -------------------------
    # FINAL SCORE
    # -------------------------

    total_score = (
        confidence_score
        +
        size_score
        +
        distance_score
        +
        target_bonus
    )

    return total_score

# -------------------------
# DECLARATIONS
# -------------------------

previous_frame = None
position_history = []


direction = "STATIONARY"

speed = 0
smoothed_speed = 0

movement_status = "STATIONARY"
no_motion_frames = 0

# -------------------------
# TARGET SYSTEM
# -------------------------

target_locked = False
target_id = 1
yolo_target_id = None
target_switch_requested = False
target_lost_frames = 0

hud_enabled = True
# -------------------------
# AUTOMATIC TARGET SWITCHING
# -------------------------

target_switch_score_margin = 15
target_switch_confirmations = 3
target_switch_counter = 0


tracked_position = None
tracked_box = None


target_velocity_x = 0
target_velocity_y = 0

# MOVEMENT VECTOR
vector_scale = 8

# PREDICTED TARGET POSITION
prediction_marker_size = 6

maximum_target_jump = 120
position_smoothing = 0.7

# Target Scoring
target_width = 0
target_height = 0

# TARGET DISTANCE
target_distance = 0
target_distance_status = "UNKNOWN"

#Distance Trend
target_distance_previous = 0
target_distance_trend = "STABLE"

#TRACKING TIMING 
target_tracking_time = 0
target_tracking_start = None

#--------------------------
#PREDICTION
#--------------------------
tracking_prediction_frames = 0
maximum_prediction_frames = 15
prediction_velocity_decay = 0.8

#Direction consistency
previous_direction_x = 0
previous_direction_y = 0

target_confidence = 0
prediction_confidence = 0


#Velocity consistency
previous_target_velocity_x = 0
previous_target_velocity_y = 0

# TARGET ACCELERATION
target_acceleration_x = 0
target_acceleration_y = 0
target_acceleration = 0
previous_target_speed = 0

# TARGET MOVEMENT STATE
target_movement_state = "STATIONARY"

# ==================================================
# HUD LAYOUT
# ==================================================

HUD_LEFT_X = 20

# TOP HEADER
HUD_TOP_Y = 30

# MAIN STATUS
HUD_STATUS_X = 20
HUD_STATUS_Y = 150
HUD_STATUS_SPACING = 23

# SYSTEM PANEL
HUD_SYSTEM_X = 450
HUD_SYSTEM_Y = 30
HUD_SYSTEM_SPACING = 20

# BOTTOM OBJECT PANEL
HUD_OBJECT_X = 20
HUD_OBJECT_Y = 350

# BOTTOM TARGET PANEL
HUD_TARGET_X = 390
HUD_TARGET_Y = 350

HUD_PANEL_SPACING = 18

# -------------------------
# YOLO FEATURE
# -------------------------
yolo = YOLOSystem()


# -------------------------
# START ARACHNE
# -------------------------

print("ARACHNE V0.1")
print("Starting camera...")


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not camera.isOpened():

    print("ERROR: Could not access the camera.")

    exit()


print("Camera ONLINE")


# -------------------------
# FPS VARIABLES
# -------------------------

fps = 0

fps_start_time = time.time()

fps_frame_count = 0


# -------------------------
# MAIN LOOP
# -------------------------

while True:

    success, frame = camera.read()


    if not success:

        print("ERROR: Could not read camera frame.")

        break

    # ==================================================
    # YOLO FEATURE
    # ==================================================

    yolo.run(frame)

    yolo_detections = yolo.get_detections()

    # -------------------------
    # YOLO TARGET SCORES
    # -------------------------

    for detection in yolo_detections:

        detection["target_score"] = (
            calculate_yolo_target_score(
                detection,
                tracked_position,
                yolo_target_id
            )
        )
        # -------------------------
        # YOLO TARGET LOSS RECOVERY
        # -------------------------

        yolo_target_visible = False

        if yolo_target_id is not None:

            for detection in yolo_detections:

                if detection["object_id"] == yolo_target_id:

                    yolo_target_visible = True
                    break


        if (
            target_locked
            and
            yolo_target_id is not None
            and
            not yolo_target_visible
        ):

            print(
                f"YOLO TARGET LOST → "
                f"ID {yolo_target_id}"
            )

            yolo_target_id = None

            target_switch_counter = 0
# -------------------------
# AUTOMATIC TARGET PRIORITY
# -------------------------

    if yolo_detections and target_locked:

        best_target = max(
            yolo_detections,
            key=lambda detection: detection["target_score"]
        )
    

        current_target = None

        for detection in yolo_detections:

            if detection["object_id"] == yolo_target_id:
                current_target = detection

                break

        if current_target is not None:

            current_score = current_target["target_score"]
            best_score = best_target["target_score"]

            score_difference = best_score - current_score

            if (
                best_target["object_id"] != yolo_target_id
                and
                score_difference >= target_switch_score_margin
            ):

                target_switch_counter += 1

            else:

                target_switch_counter = 0

            if target_switch_counter >= target_switch_confirmations:

                yolo_target_id = best_target["object_id"]

                x1, y1, x2, y2 = best_target["box"]

                tracked_position = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                tracked_box = (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )

                target_width = x2 - x1
                target_height = y2 - y1

                target_lost_frames = 0
                tracking_prediction_frames = 0

                target_switch_counter = 0

                print(
                    f"AUTO TARGET SWITCH → "
                    f"YOLO ID {yolo_target_id}"
                )

    # -------------------------
    # TARGET SWITCHING
    # -------------------------

    if target_switch_requested:

        if len(yolo_detections) > 1:

            available_targets = []

            for detection in yolo_detections:

                if detection["object_id"] != yolo_target_id:

                    available_targets.append(detection)

            if available_targets:

                next_target = max(
                    available_targets,
                    key=lambda detection: detection["confidence"]
                )

                yolo_target_id = next_target["object_id"]

                x1, y1, x2, y2 = next_target["box"]

                tracked_position = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                tracked_box = (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )

                target_width = x2 - x1
                target_height = y2 - y1

                target_lost_frames = 0
                tracking_prediction_frames = 0

                print(
                    f"TARGET SWITCHED → YOLO ID "
                    f"{yolo_target_id}"
                )

        target_switch_requested = False

        
    # -------------------------
    # YOLO TARGET SELECTION
    # -------------------------

    if yolo_detections:

        best_detection = max(
            yolo_detections,
            key=lambda detection: detection["confidence"]
        )

        yolo_target_box = best_detection["box"]
        yolo_target_class = best_detection["class_name"]
        yolo_target_confidence = best_detection["confidence"]

    else:

        yolo_target_box = None
        yolo_target_class = None
        yolo_target_confidence = 0

    # -------------------------
    # YOLO → TARGET ASSOCIATION
    # -------------------------

    yolo_associated_target = None

    if target_locked and tracked_position is not None:

        # -------------------------
        # TRY EXISTING YOLO TARGET
        # -------------------------

        if yolo_target_id is not None:

            for detection in yolo_detections:

                if detection["object_id"] == yolo_target_id:

                    yolo_associated_target = detection

                    break

        # -------------------------
        # FIND NEW YOLO TARGET
        # -------------------------

        if yolo_associated_target is None:

            closest_yolo_distance = 150

            for detection in yolo_detections:

                x1, y1, x2, y2 = detection["box"]

                yolo_center_x = (x1 + x2) // 2
                yolo_center_y = (y1 + y2) // 2

                distance = (
                    (
                        yolo_center_x
                        -
                        tracked_position[0]
                    ) ** 2
                    +
                    (
                        yolo_center_y
                        -
                        tracked_position[1]
                    ) ** 2
                ) ** 0.5

                if distance < closest_yolo_distance:

                    closest_yolo_distance = distance

                    yolo_associated_target = detection

            # Remember the newly associated target

            if yolo_associated_target is not None:

                yolo_target_id = (
                    yolo_associated_target["object_id"]
                )

        # -------------------------
        # YOLO ASSOCIATION DISPLAY
        # -------------------------
    if hud_enabled:

        if yolo_associated_target is not None:

            associated_id = yolo_associated_target["object_id"]

            associated_class = yolo_associated_target["class_name"]

            associated_confidence = (
                yolo_associated_target["confidence"]
            )

            cv2.putText(
                frame,
                f"LOCKED YOLO: ID {associated_id}",
                (350, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"CLASS: {associated_class}",
                (350, 415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"CONF: {associated_confidence:.2f}",
                (350, 440),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "LOCKED YOLO: NONE",
                (350, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )
    # ==================================================
    # MOTION DETECTION
    # ==================================================

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    gray_frame = cv2.GaussianBlur(
        gray_frame,
        (21, 21),
        0
    )


    motion_detected = False

    largest_motion_area = 0

    largest_motion_box = None

    motion_candidates = []

    selected_candidate = None

    # -------------------------
    # COMPARE FRAMES
    # -------------------------

    if previous_frame is not None:

        difference = cv2.absdiff(
            previous_frame,
            gray_frame
        )


        _, threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )


        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # -------------------------
        # FIND LARGEST MOTION
        # -------------------------

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 3000:

                x, y, w, h = cv2.boundingRect(contour)

                center_x = x + w // 2
                center_y = y + h // 2

                motion_candidates.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        center_x,
                        center_y,
                        area
                    )
                )

                if area > largest_motion_area:

                    largest_motion_area = area

                    largest_motion_box = (
                        x,
                        y,
                        w,
                        h
                    )


        # ==================================================
        # MOTION FOUND
        # ==================================================

        if len(motion_candidates) > 0:


            # -------------------------
            # ACQUIRE TARGET
            # -------------------------

            if not target_locked:

                selected_candidate = max(
                    motion_candidates,
                    key=lambda candidate: candidate[6]
                )

            # -------------------------
            # TRACK TARGET
            # -------------------------

            else:

                closest_distance = maximum_target_jump
                best_score = float("inf")

                for candidate in motion_candidates:

                    candidate_x = candidate[4]
                    candidate_y = candidate[5]

                    candidate_width = candidate[2]
                    candidate_height = candidate[3]


                    # -------------------------
                    # DISTANCE
                    # -------------------------

                    distance = (
                        (
                            candidate_x - tracked_position[0]
                        ) ** 2
                        +
                        (
                            candidate_y - tracked_position[1]
                        ) ** 2
                    ) ** 0.5


                    if distance > maximum_target_jump:

                        continue


                    # -------------------------
                    # SIZE DIFFERENCE
                    # -------------------------

                    width_difference = abs(
                        candidate_width - target_width
                    )

                    height_difference = abs(
                        candidate_height - target_height
                    )


                    # -------------------------
                    # DIRECTION DIFFERENCE
                    # -------------------------

                    candidate_dx = (
                        candidate_x
                        -
                        tracked_position[0]
                    )

                    candidate_dy = (
                        candidate_y
                        -
                        tracked_position[1]
                    )

                    direction_difference = (
                        abs(
                            candidate_dx
                            -
                            previous_direction_x
                        )
                        +
                        abs(
                            candidate_dy
                            -
                            previous_direction_y
                        )
                    )
                    candidate_velocity_x = candidate_dx
                    candidate_velocity_y = candidate_dy

                    velocity_difference = (
                        abs(
                            candidate_velocity_x
                            -
                            previous_target_velocity_x
                        )
                        +
                        abs(
                            candidate_velocity_y
                            -
                            previous_target_velocity_y
                        )
                    )


                    # -------------------------
                    # COMBINED SCORE
                    # -------------------------

                    score = (
                        distance
                        +
                        (width_difference * 0.5)
                        +
                        (height_difference * 0.5)
                        +
                        (direction_difference * 0.3)
                        +
                        (velocity_difference * 0.4)
                    )


                    if score < best_score:

                        best_score = score

                        selected_candidate = candidate


            # -------------------------
            # TARGET CONFIDENCE
            # -------------------------



            if selected_candidate is not None:

                if target_locked and 'best_score' in locals():

                    target_confidence = int(
                        100 /
                        (
                            1 +
                            (best_score / 50)
                        )
                    )

                else:

                    target_confidence = 100


        # -------------------------
        # TARGET FOUND
        # -------------------------

        if selected_candidate is not None:

            motion_detected = True
            target_locked = True
            target_lost_frames = 0

            x, y, w, h, center_x, center_y, area = selected_candidate

            # -------------------------
            # TRACKING TIMER
            # -------------------------

            if target_tracking_start is None:
                target_tracking_start = time.time()

            # -------------------------
            # SMOOTH TARGET POSITION
            # -------------------------

            if tracked_position is None:

                tracked_position = (center_x, center_y)

                target_velocity_x = 0
                target_velocity_y = 0

            else:

                old_x, old_y = tracked_position

                smooth_x = int(
                    old_x
                    + (center_x - old_x) * position_smoothing
                )

                smooth_y = int(
                    old_y
                    + (center_y - old_y) * position_smoothing
                )

                target_velocity_x = smooth_x - old_x
                target_velocity_y = smooth_y - old_y

                tracked_position = (smooth_x, smooth_y)

            # -------------------------
            # CONSISTENT BOUNDING BOX
            # -------------------------

            smooth_x, smooth_y = tracked_position

            tracked_box = (
                smooth_x - w // 2,
                smooth_y - h // 2,
                w,
                h
            )

            target_width = w
            target_height = h

            # -------------------------
            # TARGET DISTANCE
            # -------------------------

            if target_height > 0:

                target_distance = 1000 / target_height

                if target_distance < 3:
                    target_distance_status = "NEAR"
                elif target_distance < 7:
                    target_distance_status = "MEDIUM"
                else:
                    target_distance_status = "FAR"

                # -------------------------
                # DISTANCE TREND
                # -------------------------

                if target_distance_previous > 0:

                    distance_difference = (
                        target_distance - target_distance_previous
                    )

                    if distance_difference < -0.15:
                        target_distance_trend = "APPROACHING"
                    elif distance_difference > 0.15:
                        target_distance_trend = "MOVING AWAY"
                    else:
                        target_distance_trend = "STABLE"

                target_distance_previous = target_distance

            # -------------------------
            # SAVE SMOOTHED POSITION
            # -------------------------

            current_position = tracked_position

            # -------------------------
            # MOVEMENT ANALYSIS
            # -------------------------

            if len(position_history) > 0:

                previous_x, previous_y = position_history[-1]

                dx = current_position[0] - previous_x
                dy = current_position[1] - previous_y

                previous_direction_x = dx
                previous_direction_y = dy

                # Speed
                speed = (dx ** 2 + dy ** 2) ** 0.5

                # Smooth speed
                previous_target_speed = smoothed_speed

                smoothed_speed = (
                    smoothed_speed * 0.5
                    + speed * 0.5
                )

                # Acceleration
                target_acceleration = (
                    smoothed_speed - previous_target_speed
                )

                # Movement state
                if smoothed_speed < 2:
                    target_movement_state = "STATIONARY"
                elif target_acceleration > 0.5:
                    target_movement_state = "ACCELERATING"
                elif target_acceleration < -0.5:
                    target_movement_state = "DECELERATING"
                else:
                    target_movement_state = "MOVING"

                # Direction
                if abs(dx) > abs(dy):

                    if dx > 5:
                        direction = "RIGHT"
                    elif dx < -5:
                        direction = "LEFT"

                else:

                    if dy > 5:
                        direction = "DOWN"
                    elif dy < -5:
                        direction = "UP"

                # Movement classification
                if smoothed_speed < 2:
                    movement_status = "STATIONARY"
                elif smoothed_speed < 6:
                    movement_status = "SLOW"
                elif smoothed_speed < 12:
                    movement_status = "MEDIUM"
                else:
                    movement_status = "FAST"

            # -------------------------
            # SAVE POSITION HISTORY
            # -------------------------

            position_history.append(current_position)

            if len(position_history) > 20:
                position_history.pop(0)

            # ==================================================
            # YOLO VISUALISATION
            # ==================================================
        if hud_enabled:

            for detection in yolo_detections:

                x1, y1, x2, y2 = detection["box"]

                class_name = detection["class_name"]

                confidence = detection["confidence"]

                object_id = detection["object_id"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                target_score = detection["target_score"]

                yolo_label = (
                    f"ID {object_id} | "
                    f"{class_name} "
                    f"{confidence:.2f} "
                    f"S:{target_score:.1f}"
                )

                cv2.putText(
                    frame,
                    yolo_label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
            # ==================================================
            # MOTION VISUALIZATION
            # ==================================================

            # Motion box

            if hud_enabled and tracked_box is not None:

                box_x, box_y, box_w, box_h = tracked_box

                cv2.rectangle(
                    frame,
                    (box_x, box_y),
                    (box_x + box_w, box_y + box_h),
                    (255, 255, 255),
                    2
                )


            
                # ==================================================
                # TARGET LOCK RETICLE
                # ==================================================

                if hud_enabled and target_locked and tracked_box is not None:
                    box_x, box_y, box_w, box_h = tracked_box
                    corner_length = 15

                    # TOP LEFT
                    cv2.line(
                        frame,
                        (box_x, box_y),
                        (box_x + corner_length, box_y),
                        (255, 255, 255),
                        2
                    )
                    cv2.line(
                        frame,
                        (box_x, box_y),
                        (box_x, box_y + corner_length),
                        (255, 255, 255),
                        2
                    )

                    # TOP RIGHT
                    cv2.line(
                        frame,
                        (box_x + box_w, box_y),
                        (box_x + box_w - corner_length, box_y),
                        (255, 255, 255),
                        2
                    )
                    cv2.line(
                        frame,
                        (box_x + box_w, box_y),
                        (box_x + box_w, box_y + corner_length),
                        (255, 255, 255),
                        2
                    )

                    # BOTTOM LEFT
                    cv2.line(
                        frame,
                        (box_x, box_y + box_h),
                        (box_x + corner_length, box_y + box_h),
                        (255, 255, 255),
                        2
                    )
                    cv2.line(
                        frame,
                        (box_x, box_y + box_h),
                        (box_x, box_y + box_h - corner_length),
                        (255, 255, 255),
                        2
                    )

                    # BOTTOM RIGHT
                    cv2.line(
                        frame,
                        (box_x + box_w, box_y + box_h),
                        (box_x + box_w - corner_length, box_y + box_h),
                        (255, 255, 255),
                        2
                    )
                    cv2.line(
                        frame,
                        (box_x + box_w, box_y + box_h),
                        (box_x + box_w, box_y + box_h - corner_length),
                        (255, 255, 255),
                        2
                    )

                    # TARGET CENTRE
                    if tracked_position is not None:
                        cv2.circle(
                            frame,
                            (int(tracked_position[0]), int(tracked_position[1])),
                            5,
                            (255, 255, 255),
                            -1
                        )

    # ==================================================
    # MOVEMENT TRAIL
    # ==================================================

    if hud_enabled and len(position_history) > 1:

        trail_points = len(position_history)

        for i in range(1, trail_points):

            if movement_status == "STATIONARY":

                fade_ratio = i / trail_points

                thickness = max(
                    1,
                    int(2 * fade_ratio)
                )

            else:

                thickness = 2

            cv2.line(
                frame,
                position_history[i - 1],
                position_history[i],
                (255, 255, 255),
                thickness
            )

    # ==================================================
    # PROJECTED MOVEMENT VECTOR
    # ==================================================

    if ( hud_enabled and target_locked and tracked_position is not None):

        vector_start_x = tracked_position[0]
        vector_start_y = tracked_position[1]

        vector_end_x = int(
            vector_start_x
            +
            (
                target_velocity_x
                *
                vector_scale
            )
        )

        vector_end_y = int(
            vector_start_y
            +
            (
                target_velocity_y
                *
                vector_scale
            )
        )

        vector_end_x = max(
            0,
            min(639, vector_end_x)
        )

        vector_end_y = max(
            0,
            min(479, vector_end_y)
        )

        cv2.arrowedLine(
            frame,
            (
                vector_start_x,
                vector_start_y
            ),
            (
                vector_end_x,
                vector_end_y
            ),
            (255, 255, 255),
            2,
            tipLength=0.25
        )

    # ==================================================
    # NO MOTION / TARGET LOSS
    # ==================================================

    if motion_detected:

        no_motion_frames = 0
        prediction_confidence = 100
        target_lost_frames = 0
        tracking_prediction_frames = 0

    else:

        no_motion_frames += 1

        target_lost_frames += 1

            # Gradually reduce speed

        smoothed_speed *= 0.95
    

            # -------------------------
            # TARGET PERSISTENCE
            # -------------------------

        if target_locked and tracked_position is not None:

                tracking_prediction_frames += 1

                target_velocity_x *= prediction_velocity_decay

                target_velocity_y *= prediction_velocity_decay


                if no_motion_frames > 15:

                    smoothed_speed = 0

                    movement_status = "STATIONARY"

                    direction = "STATIONARY"


        # Lose target after 15 frames

        if tracking_prediction_frames > maximum_prediction_frames:

            target_locked = False

            tracked_position = None
            tracked_box = None

            target_velocity_x = 0
            target_velocity_y = 0

            tracking_prediction_frames = 0

            target_tracking_start = None
            target_tracking_time = 0


# ==================================================
# TARGET PREDICTION
# ==================================================

        if not motion_detected and target_locked:

            if tracking_prediction_frames <= maximum_prediction_frames:

                tracked_x, tracked_y = tracked_position

                predicted_x = (
                    tracked_x +
                    target_velocity_x
                )

                predicted_y = (
                    tracked_y +
                    target_velocity_y
                )

                predicted_x = max(
                    0,
                    min(639, predicted_x)
                )

                predicted_y = max(
                    0,
                    min(479, predicted_y)
                )

                tracked_position = (
                    int(predicted_x),
                    int(predicted_y)
                )
            prediction_confidence = int(
        max(
            0,
            100 - (
                tracking_prediction_frames
                *
                6
            )
        )
    )

    #TARGET PREDICTION HUD TEXT

    if (hud_enabled 
    and 
    not motion_detected and target_locked
    ):

        cv2.putText(
            frame,
            "TARGET PREDICTING",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Save current frame

    previous_frame = gray_frame


    # ==================================================
    # FPS
    # ==================================================

    fps_frame_count += 1


    current_time = time.time()


    if current_time - fps_start_time >= 1:

        fps = fps_frame_count

        fps_frame_count = 0

        fps_start_time = current_time


    if target_locked and target_tracking_start is not None:

        target_tracking_time = (
            time.time() - target_tracking_start
        )

    
    # ==================================================
    # ARACHNE HUD
    # ==================================================
    if hud_enabled:
        #TITLE

        cv2.putText(
            frame,
            "ARACHNE V0.1",
            (HUD_LEFT_X, HUD_TOP_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (HUD_LEFT_X, HUD_TOP_Y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "VISION: ONLINE",
            (HUD_LEFT_X, HUD_TOP_Y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SYSTEM: NORMAL",
            (HUD_LEFT_X, HUD_TOP_Y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ==================================================
        # TRACKING STATUS
        # ==================================================

        status_y = HUD_STATUS_Y

        if motion_detected:
            motion_text = "MOTION: DETECTED"
        else:
            motion_text = "MOTION: CLEAR"

        cv2.putText(
            frame,
            motion_text,
            (HUD_STATUS_X, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        status_y += HUD_STATUS_SPACING

        cv2.putText(
            frame,
            f"DIRECTION: {direction}",
            (HUD_STATUS_X, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        status_y += HUD_STATUS_SPACING

        cv2.putText(
            frame,
            f"SPEED: {smoothed_speed:.1f} PX/FRAME",
            (HUD_STATUS_X, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        status_y += HUD_STATUS_SPACING

        cv2.putText(
            frame,
            f"MOVEMENT: {movement_status}",
            (HUD_STATUS_X, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        status_y += HUD_STATUS_SPACING

        if target_locked:
            target_text = f"TARGET {target_id}: LOCKED"
        else:
            target_text = "TARGET: SEARCHING"

        cv2.putText(
            frame,
            target_text,
            (HUD_STATUS_X, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ==================================================
        # TARGETING RETICLE
        # ==================================================
    if hud_enabled:
        
        center_x = 320
        center_y = 240

        box_size = 80

        left = center_x - box_size
        right = center_x + box_size

        top = center_y - box_size
        bottom = center_y + box_size

        corner_length = 20


        # -------------------------
        # TOP LEFT
        # -------------------------

        cv2.line(
            frame,
            (left, top),
            (left + corner_length, top),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (left, top),
            (left, top + corner_length),
            (255, 255, 255),
            2
        )


        # -------------------------
        # TOP RIGHT
        # -------------------------

        cv2.line(
            frame,
            (right, top),
            (right - corner_length, top),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (right, top),
            (right, top + corner_length),
            (255, 255, 255),
            2
        )


        # -------------------------
        # BOTTOM LEFT
        # -------------------------

        cv2.line(
            frame,
            (left, bottom),
            (left + corner_length, bottom),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (left, bottom),
            (left, bottom - corner_length),
            (255, 255, 255),
            2
        )


        # -------------------------
        # BOTTOM RIGHT
        # -------------------------

        cv2.line(
            frame,
            (right, bottom),
            (right - corner_length, bottom),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (right, bottom),
            (right, bottom - corner_length),
            (255, 255, 255),
            2
        )


        # -------------------------
        # CENTER CROSSHAIR
        # -------------------------

        cv2.line(
            frame,
            (center_x - 10, center_y),
            (center_x + 10, center_y),
            (255, 255, 255),
            1
        )

        cv2.line(
            frame,
            (center_x, center_y - 10),
            (center_x, center_y + 10),
            (255, 255, 255),
            1
        )


        # ==================================================
        # INFORMATION PANEL
        # ==================================================
    if hud_enabled:

        panel_x = 450
        panel_y = 30


        cv2.putText(
            frame,
            "ARACHNE SYSTEM",
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            "CAMERA   ONLINE",
            (panel_x, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            "VISION   READY",
            (panel_x, panel_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            "AI       STANDBY",
            (panel_x, panel_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
        frame,
        f"ACCEL   {target_acceleration:.1f}",
        (panel_x, panel_y + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
        )
        cv2.putText(
            frame,
            f"STATE   {target_movement_state}",
            (panel_x, panel_y + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ==================================================
        # MULTI-TARGET INFORMATION PANEL
        # ==================================================
    if hud_enabled:

        panel_x = HUD_OBJECT_X
        panel_y = HUD_OBJECT_Y

        cv2.putText(
            frame,
            "VISION OBJECTS",
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        panel_y += HUD_PANEL_SPACING

        for detection in yolo_detections:

            object_id = detection["object_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            target_marker = ""

            if object_id == yolo_target_id:

                target_marker = " <TARGET>"

            object_text = (
                f"ID {object_id} | "
                f"{class_name.upper()} | "
                f"{confidence:.2f}"
                f"{target_marker}"
            )

            cv2.putText(
                frame,
                object_text,
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )

            panel_y += HUD_PANEL_SPACING
            
        # ==================================================
        # TARGET ANALYTICS PANEL
        # ==================================================
    if hud_enabled:

        analytics_x = HUD_TARGET_X
        analytics_y = HUD_TARGET_Y

        cv2.putText(
            frame,
            "TARGET ANALYTICS",
            (analytics_x, analytics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        analytics_y += HUD_PANEL_SPACING

        if target_locked:

            target_class_display = "UNKNOWN"
            target_confidence_display = 0

            if yolo_associated_target is not None:

                target_class_display = (
                    yolo_associated_target["class_name"]
                )

                target_confidence_display = (
                    yolo_associated_target["confidence"]
                )

            analytics_lines = [

                f"ID: {yolo_target_id}",

                f"CLASS: "
                f"{target_class_display.upper()}",

                f"CONF: "
                f"{target_confidence_display:.2f}",

                f"DIST: "
                f"{target_distance:.1f}",

                f"TREND: "
                f"{target_distance_trend}",

                f"SPEED: "
                f"{smoothed_speed:.1f}",

                f"DIRECTION: "
                f"{direction}",

                f"STATE: "
                f"{target_movement_state}",

                f"ACCEL: "
                f"{target_acceleration:.1f}",

                f"TRACK: "
                f"{target_tracking_time:.1f}s"
            ]

            for line in analytics_lines:

                cv2.putText(
                    frame,
                    line,
                    (analytics_x, analytics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 0, 0),
                    1
                )

                analytics_y += HUD_PANEL_SPACING

        else:

            cv2.putText(
                frame,
                "NO TARGET",
                (analytics_x, analytics_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(
        "ARACHNE V0.1",
        frame
    )


    # -------------------------
    # QUIT
    # -------------------------

    key = cv2.waitKey(1) & 0xFF

    # -------------------------
    # TARGET SWITCH
    # -------------------------

    if key == ord("t"):

        target_switch_requested = True

    #-----------------------------
    #Enable/Disable HUD
    #----------------------------
    if key == ord("h"):
        hud_enabled = not hud_enabled

        if hud_enabled:
            print("HUD: ENABLED")
        else:
            print("HUD: DISABLED")

    # -------------------------
    # QUIT
    # -------------------------

    if key == ord("q"):

        break


# ==================================================
# SHUTDOWN
# ==================================================

camera.release()

cv2.destroyAllWindows()

print("\nARACHNE OFFLINE.")