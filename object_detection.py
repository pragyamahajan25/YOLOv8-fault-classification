# object_detection.py

import os
from pathlib import Path
import json
import cv2  # OpenCV for saving images
from ultralytics import YOLO

# Create output directories
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

categories = ["no_fault", "blur_extreme", "blur_medium", "blur_low"]
for category in categories:
    (output_dir / category).mkdir(exist_ok=True)

# Input folders for image data
input_folders = {
    "no_fault": r"D:\Users\Pragya Mahajan\Documents\Arbeit\Phase 2\Original_dataset_faults\NoFault",
    "blur_extreme": r"D:\Users\Pragya Mahajan\Documents\Arbeit\Phase 2\Original_dataset_faults\Blur\e",
    "blur_low": r"D:\Users\Pragya Mahajan\Documents\Arbeit\Phase 2\Original_dataset_faults\Blur\l",
    "blur_medium": r"D:\Users\Pragya Mahajan\Documents\Arbeit\Phase 2\Original_dataset_faults\Blur\m",
}

# Function to check if a folder has images
def check_folders(input_folders):
    for category, folder_path in input_folders.items():
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Error: Folder '{folder_path}' does not exist!")
            continue

        # Check for image files (jpg, png)
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not images:
            print(f"Warning: No image files found in '{folder_path}'.")
        else:
            print(f"'{folder_path}' contains {len(images)} image(s).")

# Check folders for images
check_folders(input_folders)

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  

# Prepare JSON output data
detection_results = []

# Iterate through input folders
for category, folder_path in input_folders.items():
    folder = Path(folder_path)
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    for img_path in images:
        # Perform object detection
        results = model(img_path)
        result = results[0]  # Take the first result

        # Extract confidence scores and class names
        confidences = [box.conf.item() for box in result.boxes]
        classes = [model.names[int(box.cls)] for box in result.boxes]

        # Initialize data entry
        detection_entry = {"Image Name": img_path.name, "Category": category, "Detections": []}

        # Populate detection data
        for conf, cls in zip(confidences, classes):
            detection_entry["Detections"].append({"Class": cls, "Confidence": f"{conf:.2f}"})

        # Append to results
        detection_results.append(detection_entry)

        # Save processed image with bounding boxes
        processed_image = result.plot()  # Use .plot() to draw on the image

        # Ensure the output directory exists
        output_category_dir = output_dir / category
        output_category_dir.mkdir(parents=True, exist_ok=True)

        # Save the image with bounding boxes
        processed_image_path = output_category_dir / img_path.name
        processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  # Convert to BGR
        cv2.imwrite(str(processed_image_path), processed_image_bgr)  # Save using OpenCV

# Save detection results to JSON
output_json_path = output_dir / "detection_results.json"
with open(output_json_path, "w") as json_file:
    json.dump(detection_results, json_file, indent=4)

print(f"Detection results saved to {output_json_path}")
