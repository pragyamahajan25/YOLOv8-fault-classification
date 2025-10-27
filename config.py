import torch

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths for dataset
data_path = "Dataset"
train_path = f"{data_path}/Train_dataset_faults/Blur"
val_path = f"{data_path}/Val_dataset_faults/Blur"  # Assuming you now have a validation dataset
test_path = f"{data_path}/Test_dataset_faults/Blur"
