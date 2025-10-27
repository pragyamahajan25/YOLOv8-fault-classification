import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
import random

# Define data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class CustomDataset(Dataset):
    def __init__(self, folder_path, label=None, transform=None):
        self.folder_path = Path(folder_path)
        self.label = label
        self.transform = transform
        self.image_paths = list(self.folder_path.glob("*.jpg")) + list(self.folder_path.glob("*.png"))
        
        # Print statements for debugging
        print(f"Initializing dataset with {len(self.image_paths)} images from {self.folder_path}")
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {self.folder_path}")
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Debugging print statements
        #print(f"Loading image: {img_path}")
        
        return image, self.label

def create_train_val_loaders(train_path, val_path, batch_size=32, train_split=0.8, seed=42):
    # Define datasets for training and validation
    train_datasets = {
        "l": CustomDataset(f"{train_path}/l", label=0, transform=data_transforms["train"]),
        "m": CustomDataset(f"{train_path}/m", label=1, transform=data_transforms["train"]),
        "e": CustomDataset(f"{train_path}/e", label=2, transform=data_transforms["train"]),
        "no_fault": CustomDataset(f"{train_path}/No fault", label=3, transform=data_transforms["train"])
    }
    
    val_datasets = {
        "l": CustomDataset(f"{val_path}/l", label=0, transform=data_transforms["val"]),
        "m": CustomDataset(f"{val_path}/m", label=1, transform=data_transforms["val"]),
        "e": CustomDataset(f"{val_path}/e", label=2, transform=data_transforms["val"]),
        "no_fault": CustomDataset(f"{val_path}/No fault", label=3, transform=data_transforms["val"])
    }
    
        # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(list(train_datasets.values()) + list(val_datasets.values()))
    
     # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle indices of the combined dataset
    total_size = len(combined_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Subset the dataset into train and validation
    train_data = Subset(combined_dataset, train_indices)
    val_data = Subset(combined_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Debugging: Print dataset sizes
    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")

    return train_loader, val_loader

def create_test_loader(test_path, batch_size=32):
    test_datasets = {
            "l": CustomDataset(f"{test_path}/l", label=0, transform=data_transforms["test"]),
            "m": CustomDataset(f"{test_path}/m", label=1, transform=data_transforms["test"]),
            "e": CustomDataset(f"{test_path}/e", label=2, transform=data_transforms["test"]),
            "no_fault": CustomDataset(f"{test_path}/No fault", label=3, transform=data_transforms["test"])
        }
        
    combined_test_dataset = torch.utils.data.ConcatDataset(list(test_datasets.values()))
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Debugging: Print number of test samples
    print(f"Test loader initialized with {len(combined_test_dataset)} samples")
    return test_loader

