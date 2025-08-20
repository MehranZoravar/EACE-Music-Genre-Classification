import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import os

# pick one of these per run
model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=10)
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=10)


# Modify the classifier head to match the number of classes in your dataset (e.g., 10 classes)
#model.head = nn.Linear(model.head.in_features, 10)
#model.fc = nn.Linear(model.fc.in_features, 10)
model.head = nn.Linear(model.head.in_features, 10)  # Ensure 10 output classes

# Set up the device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset from the directory structure
dataset_path = 'Dataset_path'
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Define the split percentages
train_percent = 0.5
val_percent = 0.2
calibration_percent = 0.2
test_percent = 0.1

# Calculate the sizes for each dataset split
train_size = int(train_percent * len(full_dataset))
val_size = int(val_percent * len(full_dataset))
calibration_size = int(calibration_percent* len(full_dataset))
test_size = len(full_dataset) - train_size - val_size - calibration_size

# Split the dataset into training, validation, calibration, and test sets
train_dataset, val_dataset, calibration_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, calibration_size, test_size]
)

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
cal_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Training loop

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Training complete.')

"""
# Training loop for Swin transformer model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # outputs shape: [batch_size, num_classes, 7, 7]

        # Apply global average pooling to manually flatten the output
        outputs = outputs.mean(dim=[2, 1])  # Now outputs shape is [batch_size, num_classes]
        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.mean(dim=[2, 1])  # Flatten in validation loop as well
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Training complete.')
"""
# Save the trained model

model_save_dir = 'model_path'
os.makedirs(model_save_dir, exist_ok=True)  # Create the directory if it does not exist
model_save_path = os.path.join(model_save_dir, 'model_weights_deit_mel.pth')
torch.save(model.state_dict(), model_save_path)

print(f'Model saved to {model_save_path}')

### Test Loop ###
correct = 0
total = 0
softmax_outputs = []
all_labels = []  # To store labels

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        #outputs = outputs.mean(dim=[2, 1])
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Apply softmax to the outputs
        softmax_output = torch.softmax(outputs, dim=1)

        # Append the softmax outputs and labels to the lists
        softmax_outputs.append(softmax_output.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Convert lists to NumPy arrays
softmax_outputs = np.concatenate(softmax_outputs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Save the softmax outputs and labels for the test set
np.save('./softmax_outputs_test.npy', softmax_outputs)
np.save('./labels_test.npy', all_labels)

print("Test softmax outputs and labels have been saved as .npy files.")

### Calibration Loop ###
softmax_outputs_cal = []
all_labels_cal = []  # To store calibration labels

with torch.no_grad():
    for images, labels in cal_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        #outputs = outputs.mean(dim=[2, 1])
        _, predicted = torch.max(outputs, 1)

        # Apply softmax to the outputs
        softmax_output = torch.softmax(outputs, dim=1)

        # Append the softmax outputs and labels to the lists
        softmax_outputs_cal.append(softmax_output.cpu().numpy())
        all_labels_cal.append(labels.cpu().numpy())

# Convert lists to NumPy arrays
softmax_outputs_cal = np.concatenate(softmax_outputs_cal, axis=0)
all_labels_cal = np.concatenate(all_labels_cal, axis=0)

# Save the softmax outputs and labels for the calibration set
np.save('./softmax_outputs_cal.npy', softmax_outputs_cal)
np.save('./labels_cal.npy', all_labels_cal)

print("Calibration softmax outputs and labels have been saved as .npy files.")
