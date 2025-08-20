import numpy as np
import torch
from torchvision import datasets, transforms
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Dataset path
dataset_path = './mel_spectrogram'

# Define dataset split ratios
train_percent = 0.5
val_percent = 0.2
calibration_percent = 0.2
test_percent = 0.1

# Define transformations (must match your model training setup)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataset_size = len(full_dataset)

# Calculate split sizes
train_size = int(train_percent * dataset_size)
val_size = int(val_percent * dataset_size)
cal_size = int(calibration_percent * dataset_size)
test_size = dataset_size - train_size - val_size - cal_size

# Split indices reproducibly
full_indices = list(range(dataset_size))
splits = torch.utils.data.random_split(full_indices, [train_size, val_size, cal_size, test_size])
split_indices = [list(split) for split in splits]

# Save split indices
output_dir = "./output/splits"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "saved_split_indices.npy")

np.save(save_path, np.array(split_indices, dtype=object))
print(f"✅ Split indices saved to: {save_path}")
