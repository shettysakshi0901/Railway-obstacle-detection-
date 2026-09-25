from ultralytics import YOLO
import cv2

model = YOLO(
    r"C:\Users\ACER\Downloads\Railroad_and_Obstacle_detection-master\runs\detect\train\weights\best.pt"
)

video = (
    r"C:\Users\ACER\Downloads\Railroad_and_Obstacle_detection-master"
    r"\Railroad_and_Obstacle_detection-master\video\railway_video.mp4.mp4"
)

results = model.predict(
    source=video,
    conf=0.7,
    stream=True,
    verbose=False
)

# Person class in our dataset = 2
PERSON_CLASS = 2

for i, result in enumerate(results):

    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == PERSON_CLASS:

            image = result.plot()

            output = (
                r"C:\Users\ACER\Downloads\Railroad_and_Obstacle_detection-master"
                r"\runs\detect\predict-2\person_detection.jpg"
            )

            cv2.imwrite(output, image)

            print("PERSON FOUND AT FRAME:", i)
            print("Saved:", output)
            break

    else:
        continue

    break

else:
    print("NO PERSON FOUND")