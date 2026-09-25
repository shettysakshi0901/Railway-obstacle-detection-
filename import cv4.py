import cv2
import math
import time
import os
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "yolo11n.pt"

# ---------------------------------------------------------
# INPUT MODE
# ---------------------------------------------------------
# Choose:
#
# 1 = Webcam
# 2 = Video file
# 3 = Image / Screenshot
#
INPUT_MODE = 3

# If INPUT_MODE = 1
WEBCAM_INDEX = 0

# If INPUT_MODE = 2
VIDEO_SOURCE = "video 3.mp4"

# If INPUT_MODE = 3
IMAGE_SOURCE = "Screenshot 2026-09-23 144420 (1).png"


# =========================================================
# VEHICLE PARAMETERS
# =========================================================

INITIAL_SPEED_KMH = 50.0

BRAKING_DECELERATION = 7.0       # m/s²
REACTION_TIME = 0.5              # seconds


# =========================================================
# APPROXIMATE OBJECT DIMENSIONS
# =========================================================

KNOWN_WIDTH = {
    "person": 0.5,
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.5,
    "motorcycle": 0.8,
    "bicycle": 0.6,
    "dog": 0.5,
    "cat": 0.3,
}


# =========================================================
# CAMERA PARAMETERS
# =========================================================

FOCAL_LENGTH = 700


# =========================================================
# YOLO CONFIDENCE
# =========================================================

CONFIDENCE_THRESHOLD = 0.45


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# FUNCTIONS
# =========================================================

def estimate_distance(object_name, pixel_width):
    """
    Estimate approximate distance using:

        distance = (real_width * focal_length) / pixel_width

    This is only an approximate demonstration.
    """

    if pixel_width <= 0:
        return None

    real_width = KNOWN_WIDTH.get(object_name)

    if real_width is None:
        return None

    distance = (
        real_width * FOCAL_LENGTH
    ) / pixel_width

    return distance


# ---------------------------------------------------------

def calculate_ttc(distance, speed_kmh):
    """
    Calculate approximate Time To Collision (TTC).
    """

    speed_ms = speed_kmh / 3.6

    if speed_ms <= 0:
        return float("inf")

    return distance / speed_ms


# ---------------------------------------------------------

def braking_distance(speed_kmh):
    """
    Calculate total stopping distance:

        reaction distance + braking distance
    """

    speed_ms = speed_kmh / 3.6

    reaction_distance = (
        speed_ms * REACTION_TIME
    )

    braking_distance_only = (
        speed_ms ** 2
    ) / (
        2 * BRAKING_DECELERATION
    )

    total_distance = (
        reaction_distance +
        braking_distance_only
    )

    return total_distance


# ---------------------------------------------------------

def get_risk_level(ttc, distance, speed_kmh):
    """
    Determine obstacle risk.
    """

    stop_distance = braking_distance(
        speed_kmh
    )

    if (
        distance <= stop_distance
        or ttc <= 1.5
    ):
        return "CRITICAL", (0, 0, 255)

    elif ttc <= 3.0:

        return "WARNING", (0, 165, 255)

    else:

        return "SAFE", (0, 255, 0)


# ---------------------------------------------------------

def simulate_braking(initial_speed):
    """
    Software-only braking simulation.
    """

    speed = initial_speed

    dt = 0.1

    simulation = []

    while speed > 0:

        speed_ms = speed / 3.6

        speed_ms -= (
            BRAKING_DECELERATION * dt
        )

        if speed_ms < 0:
            speed_ms = 0

        speed = speed_ms * 3.6

        simulation.append(speed)

    return simulation


# ---------------------------------------------------------

def process_frame(frame, current_speed):
    """
    Run YOLO detection and draw all information
    on the frame.

    Returns:

        processed_frame
        highest_risk
        highest_color
        detected_objects
    """

    frame_height, frame_width = frame.shape[:2]

    # -----------------------------------------------------
    # YOLO DETECTION
    # -----------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    highest_risk = "SAFE"

    highest_color = (0, 255, 0)

    detected_objects = []

    risk_priority = {
        "SAFE": 0,
        "WARNING": 1,
        "CRITICAL": 2
    }

    # -----------------------------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------------------------

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Confidence
            confidence = float(
                box.conf[0]
            )

            # Class
            class_id = int(
                box.cls[0]
            )

            object_name = model.names[
                class_id
            ]

            # Pixel width
            pixel_width = x2 - x1

            # -------------------------------------------------
            # DISTANCE
            # -------------------------------------------------

            distance = estimate_distance(
                object_name,
                pixel_width
            )

            # Unknown objects
            if distance is None:
                distance = 999

            # -------------------------------------------------
            # TTC
            # -------------------------------------------------

            ttc = calculate_ttc(
                distance,
                current_speed
            )

            # -------------------------------------------------
            # RISK
            # -------------------------------------------------

            risk, color = get_risk_level(
                ttc,
                distance,
                current_speed
            )

            # -------------------------------------------------
            # HIGHEST RISK
            # -------------------------------------------------

            if (
                risk_priority[risk]
                >
                risk_priority[highest_risk]
            ):

                highest_risk = risk

                highest_color = color

            # -------------------------------------------------
            # BOUNDING BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # -------------------------------------------------
            # OBJECT LABEL
            # -------------------------------------------------

            label = (
                f"{object_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # -------------------------------------------------
            # DISTANCE TEXT
            # -------------------------------------------------

            if distance < 100:

                distance_text = (
                    f"Distance: "
                    f"{distance:.1f} m"
                )

                cv2.putText(
                    frame,
                    distance_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # -------------------------------------------------
            # TTC TEXT
            # -------------------------------------------------

            if (
                ttc != float("inf")
                and ttc < 100
            ):

                ttc_text = (
                    f"TTC: "
                    f"{ttc:.2f}s"
                )

                cv2.putText(
                    frame,
                    ttc_text,
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # -------------------------------------------------
            # STORE DETECTION
            # -------------------------------------------------

            detected_objects.append({

                "name": object_name,

                "confidence": confidence,

                "distance": distance,

                "ttc": ttc,

                "risk": risk

            })

    # =====================================================
    # DASHBOARD
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame_width, 105),
        (30, 30, 30),
        -1
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "AI ROAD & RAILWAY OBSTACLE DETECTION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -----------------------------------------------------
    # SPEED
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Speed: {current_speed:.1f} km/h",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"STATUS: {highest_risk}",
        (300, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        highest_color,
        2
    )

    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    if highest_risk == "SAFE":

        action = "CONTINUE"

    elif highest_risk == "WARNING":

        action = "WARNING - SLOW DOWN"

    else:

        action = "EMERGENCY BRAKING"

    cv2.putText(
        frame,
        action,
        (520, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        highest_color,
        2
    )

    return (
        frame,
        highest_risk,
        highest_color,
        detected_objects
    )


# =========================================================
# INPUT MODE 3: IMAGE / SCREENSHOT
# =========================================================

if INPUT_MODE == 3:

    print("\nOpening image...")

    image = cv2.imread(
        IMAGE_SOURCE
    )

    if image is None:

        print(
            "ERROR: Cannot open image:"
        )

        print(IMAGE_SOURCE)

        exit()

    print(
        "Image loaded successfully."
    )

    # Process image
    processed_frame, highest_risk, highest_color, objects = process_frame(
        image,
        INITIAL_SPEED_KMH
    )

    # -----------------------------------------------------
    # IMAGE RESULT
    # -----------------------------------------------------

    print("\nDetected objects:")

    if len(objects) == 0:

        print("No objects detected.")

    else:

        for obj in objects:

            print(
                f"{obj['name']} | "
                f"Confidence: "
                f"{obj['confidence']:.2f} | "
                f"Distance: "
                f"{obj['distance']:.2f} m | "
                f"TTC: "
                f"{obj['ttc']:.2f} s | "
                f"Risk: "
                f"{obj['risk']}"
            )

    # -----------------------------------------------------
    # SHOW IMAGE
    # -----------------------------------------------------

    cv2.imshow(
        "AI Obstacle Detection - Image",
        processed_frame
    )

    print(
        "\nPress any key to close."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    exit()


# =========================================================
# INPUT MODE 1 OR 2
# WEBCAM / VIDEO
# =========================================================

if INPUT_MODE == 1:

    print("\nStarting webcam...")

    cap = cv2.VideoCapture(
        WEBCAM_INDEX
    )

elif INPUT_MODE == 2:

    print(
        f"\nOpening video: "
        f"{VIDEO_SOURCE}"
    )

    cap = cv2.VideoCapture(
        VIDEO_SOURCE
    )

else:

    print("Invalid INPUT_MODE.")

    exit()


# =========================================================
# CHECK VIDEO / CAMERA
# =========================================================

if not cap.isOpened():

    print(
        "ERROR: Cannot open "
        "camera/video."
    )

    exit()


# =========================================================
# VARIABLES
# =========================================================

current_speed = INITIAL_SPEED_KMH

braking_active = False

braking_start_time = None


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "\nVideo/camera stream ended."
        )

        break

    # -----------------------------------------------------
    # PROCESS FRAME
    # -----------------------------------------------------

    (
        frame,
        highest_risk,
        highest_color,
        detected_objects
    ) = process_frame(
        frame,
        current_speed
    )

    # -----------------------------------------------------
    # BRAKING DECISION
    # -----------------------------------------------------

    if (
        highest_risk == "CRITICAL"
        and not braking_active
    ):

        braking_active = True

        braking_start_time = time.time()

        print(
            "\n!!! EMERGENCY BRAKING "
            "ACTIVATED !!!"
        )

        # -------------------------------------------------
        # BRAKING SIMULATION
        # -------------------------------------------------

        simulation = simulate_braking(
            current_speed
        )

        print(
            "Braking simulation:"
        )

        for i, speed in enumerate(
            simulation
        ):

            if i % 5 == 0:

                print(
                    f"Time: "
                    f"{i * 0.1:.1f}s | "
                    f"Speed: "
                    f"{speed:.1f} km/h"
                )

    # -----------------------------------------------------
    # BRAKING SPEED SIMULATION
    # -----------------------------------------------------

    if braking_active:

        elapsed = (
            time.time()
            -
            braking_start_time
        )

        speed_ms = (
            INITIAL_SPEED_KMH
            /
            3.6
        )

        speed_ms -= (
            BRAKING_DECELERATION
            *
            elapsed
        )

        if speed_ms <= 0:

            speed_ms = 0

            current_speed = 0

        else:

            current_speed = (
                speed_ms * 3.6
            )

    else:

        current_speed = (
            INITIAL_SPEED_KMH
        )

    # -----------------------------------------------------
    # VEHICLE STOPPED
    # -----------------------------------------------------

    if (
        braking_active
        and current_speed <= 0
    ):

        cv2.putText(
            frame,
            "VEHICLE STOPPED",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "AI Obstacle Detection System",
        frame
    )

    # -----------------------------------------------------
    # KEYBOARD
    # -----------------------------------------------------

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )

    # Q = quit
    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print(
    "System terminated."
)
