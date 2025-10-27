from pathlib import Path
import shutil
import torch
from model import initialize_model
from dataset import create_test_loader
from config import device, test_path

# Load the trained model
model = initialize_model(num_classes=4)
model.load_state_dict(torch.load("blur_classifier.pth"))
model = model.to(device)

# Set up test loader
test_loader = create_test_loader(test_path)

# Evaluate model on test set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Output folders for predicted images
output_folders = {
    0: Path(f"{test_path}/predicted_low"),
    1: Path(f"{test_path}/predicted_medium"),
    2: Path(f"{test_path}/predicted_extreme"),
    3: Path(f"{test_path}/predicted_no_fault"),
}
for folder in output_folders.values():
    folder.mkdir(parents=True, exist_ok=True)

# Helper function to map index to original dataset and get image path
def get_image_path(index, datasets):
    for dataset in datasets.datasets:  # Access constituent datasets
        if index < len(dataset):
            return dataset.image_paths[index]  # Return the image path
        index -= len(dataset)
    raise IndexError("Index out of range")

# Moving test images based on predicted labels
with torch.no_grad():
    for batch_idx, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i, pred in enumerate(predicted):
            global_index = batch_idx * test_loader.batch_size + i
            src_path = get_image_path(global_index, test_loader.dataset)  # Retrieve correct image path
            dest_folder = output_folders[pred.item()]
            shutil.copy(src_path, dest_folder / Path(src_path).name)
