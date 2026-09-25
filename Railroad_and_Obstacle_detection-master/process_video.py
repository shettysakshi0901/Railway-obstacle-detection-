from ultralytics import YOLO
import cv2
import time
import winsound

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(
    r"C:\Users\ACER\Downloads\Railroad_and_Obstacle_detection-master\runs\detect\train\weights\best.pt"
)

# ============================================================
# VIDEO FOLDER
# ============================================================

video_folder = (
    r"C:\Users\ACER\Downloads\Railroad_and_Obstacle_detection-master"
    r"\Railroad_and_Obstacle_detection-master"
    r"\video"
)

videos = [
    "complex_crash.mp4",
    "complex1.mp4",
    "complex2.mp4",

    
]

# ============================================================
# FUNCTION:
# CHECK WHETHER PERSON IS ON TRACK
# ============================================================

def person_is_on_track(x1, y1, x2, y2, frame_width, frame_height):

    # Use the bottom-center of the person's bounding box.
    # This represents where the person's feet touch the ground.

    person_x = int((x1 + x2) / 2)
    person_y = int(y2)

    # --------------------------------------------------------
    # TRACK DANGER AREA
    #
    # These values are normalized, so they work with different
    # video resolutions.
    #
    # IMPORTANT:
    # Adjust these four points after seeing complex1.mp4.
    # --------------------------------------------------------

    track_polygon = [
        (
            int(frame_width * 0.35),
            int(frame_height * 0.45)
        ),
        (
            int(frame_width * 0.65),
            int(frame_height * 0.45)
        ),
        (
            int(frame_width * 0.95),
            int(frame_height * 1.00)
        ),
        (
            int(frame_width * 0.05),
            int(frame_height * 1.00)
        )
    ]

    # Check whether person's feet are inside track area
    inside = cv2.pointPolygonTest(
        __import__("numpy").array(track_polygon, dtype="int32"),
        (person_x, person_y),
        False
    )

    return inside >= 0


# ============================================================
# PROCESS VIDEOS
# ============================================================

for video_name in videos:

    video_path = video_folder + "\\" + video_name

    print("\n======================================")
    print("Processing:", video_name)
    print("======================================")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open:", video_name)
        continue

    print("Video opened successfully.")

    obstacle_frames = 0
    last_warning_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            iou=0.5,
            verbose=False
        )

        result = results[0]

        annotated_frame = frame.copy()

        danger = False
        danger_type = ""

        # ====================================================
        # CHECK DETECTIONS
        # ====================================================

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes

            for i in range(len(boxes)):

                x1, y1, x2, y2 = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = float(
                    boxes.conf[i].cpu().item()
                )

                class_id = int(
                    boxes.cls[i].cpu().item()
                )

                class_name = model.names[class_id]

                # ------------------------------------------------
                # PERSON
                # ------------------------------------------------

                if class_name == "Person":

                    # Check ONLY the person's feet position
                    on_track = person_is_on_track(
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_width,
                        frame_height
                    )

                    if on_track:

                        # PERSON IS ACTUALLY ON TRACK
                        danger = True
                        danger_type = "PERSON ON RAILWAY TRACK"

                        box_color = (0, 0, 255)

                    else:

                        # PERSON IS BESIDE TRACK
                        # Do NOT generate warning.
                        box_color = (0, 255, 0)

                # ------------------------------------------------
                # TREE
                # ------------------------------------------------

                elif class_name == "Tree":

                    # A tree is considered an obstacle.
                    danger = True
                    danger_type = "FALLEN TREE"
                    box_color = (0, 0, 255)

                # ------------------------------------------------
                # OTHER OBJECTS
                # ------------------------------------------------

                elif class_name in ["Animal", "Rock", "Trash", "Car"]:

                    danger = True
                    danger_type = class_name.upper()
                    box_color = (0, 0, 255)

                else:

                    # Ignore unknown classes
                    continue

                # =================================================
                # DRAW BOX
                # =================================================

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

        # ====================================================
        # CONFIRM DANGER FOR 3 FRAMES
        # ====================================================

        if danger:

            obstacle_frames += 1

        else:

            obstacle_frames = 0

        # ====================================================
        # EMERGENCY WARNING
        # ====================================================

        if obstacle_frames >= 3:

            # Large red border
            cv2.rectangle(
                annotated_frame,
                (5, 5),
                (
                    frame_width - 5,
                    frame_height - 5
                ),
                (0, 0, 255),
                12
            )

            # Large warning panel
            cv2.rectangle(
                annotated_frame,
                (20, 20),
                (
                    frame_width - 20,
                    190
                ),
                (0, 0, 0),
                -1
            )

            # DANGER
            cv2.putText(
                annotated_frame,
                "!!! DANGER !!!",
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 0, 255),
                5
            )

            # Object
            cv2.putText(
                annotated_frame,
                danger_type,
                (50, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            # Stop message
            cv2.putText(
                annotated_frame,
                "DO NOT GO FORWARD!",
                (50, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            # Braking simulation
            cv2.putText(
                annotated_frame,
                "EMERGENCY BRAKING ACTIVATED",
                (40, frame_height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            # =================================================
            # WARNING SOUND
            # =================================================

            current_time = time.time()

            if current_time - last_warning_time > 1.5:

                winsound.Beep(1200, 500)

                last_warning_time = current_time

        else:

            cv2.putText(
                annotated_frame,
                "ROAD / RAILWAY CLEAR",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # ====================================================
        # SHOW VIDEO
        # ====================================================

        cv2.imshow(
            "AI ROAD & RAILWAY OBSTACLE DETECTION",
            annotated_frame
        )

        # Q = stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()

print("\nAll selected videos processed.")