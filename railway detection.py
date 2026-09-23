# Project Notes

## Dataset

Source: Roboflow Railway Track Obstacle Detection dataset.

Classes:

1. Animal
2. Car
3. Person
4. Rock
5. Trash
6. Tree

## Training

YOLOv8 Nano was trained for 20 epochs with an image size of 640 x 640.

The trained checkpoint used for inference is `best.pt`.

## Model evaluation recorded during the project

Overall validation results:

- Precision: approximately 0.74
- Recall: approximately 0.604
- mAP@50: approximately 0.722
- mAP@50-95: approximately 0.410

Person class:

- Precision: approximately 0.925
- Recall: approximately 0.772
- mAP@50: approximately 0.900

These are validation results from the project training run and should not be interpreted as a guarantee of real-world railway safety.

## Project concept

The system combines object detection with a spatial danger decision. A person detected outside the railway track should not automatically trigger a railway intrusion alert. A person inside the defined track danger region should trigger the warning/braking simulation.

The current project does not control real braking hardware.
