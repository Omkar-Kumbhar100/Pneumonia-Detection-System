"""
train.py – Train a pneumonia classifier using ResNet18 transfer learning.
Dataset expected at: data/chest_xray/train , data/chest_xray/val
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ------------------- 1. Setup Device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- 2. Data Paths -------------------
train_path = "data/chest_xray/train"
val_path   = "data/chest_xray/val"

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training folder not found at {train_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation folder not found at {val_path}")

# ------------------- 3. Transforms -------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------- 4. Datasets & Loaders -------------------
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_path,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

# ------------------- 5. Model (Transfer Learning) -------------------
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------- 6. Training Loop -------------------
num_epochs = 5
train_losses = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
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

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Val Acc: {val_acc:.4f}")

# ------------------- 7. Save Model -------------------
torch.save(model.state_dict(), "model.pth")
print("\n✅ Model saved as model.pth")

# ------------------- 8. Plot & Save Figure -------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(val_accuracies, marker='o', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png')
plt.show()
print("✅ Training plot saved as training_plot.png")