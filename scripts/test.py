import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import random

# Set seed
random.seed(42)
torch.manual_seed(42)

# Device setup
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# Load model
model = timm.create_model('deit_base_patch16_224', pretrained=True)
#model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
model.head = nn.Linear(model.num_features, 10)

# Load weights
model_save_path = '../output/model_weights_deit_mel.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model = model.to(device)
model.eval()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset_path = './mel_spectrogram'
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Load saved split indices
split_indices = np.load('../saved_split_indices.npy', allow_pickle=True)
train_idx, val_idx, cal_idx, test_idx = split_indices

# Subsets
calibration_dataset = Subset(full_dataset, cal_idx)
test_dataset = Subset(full_dataset, test_idx)

# DataLoaders
cal_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

### Test Loop ###
correct = 0
total = 0
softmax_outputs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        #outputs = outputs.mean(dim=[2, 1])
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        softmax_output = torch.softmax(outputs, dim=1)
        softmax_outputs.append(softmax_output.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Save test outputs
softmax_outputs = np.concatenate(softmax_outputs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
np.save('../output/DeiT_model/softmax_outputs_test.npy', softmax_outputs)
np.save('../output/DeiT_model/labels_test.npy', all_labels)
print("Saved test outputs.")

### Calibration Loop ###
softmax_outputs_cal = []
all_labels_cal = []

with torch.no_grad():
    for images, labels in cal_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        #outputs = outputs.mean(dim=[2, 1])
        softmax_output = torch.softmax(outputs, dim=1)
        softmax_outputs_cal.append(softmax_output.cpu().numpy())
        all_labels_cal.append(labels.cpu().numpy())

# Save calibration outputs
softmax_outputs_cal = np.concatenate(softmax_outputs_cal, axis=0)
all_labels_cal = np.concatenate(all_labels_cal, axis=0)
np.save('../output/DeiT_model/softmax_outputs_cal.npy', softmax_outputs_cal)
np.save('../output/DeiT_model/labels_cal.npy', all_labels_cal)
print("Saved calibration outputs.")
