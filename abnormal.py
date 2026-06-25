import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# ==========================
# CONFIG
# ==========================

BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_CLASSES = 2   # binary
NUM_EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==========================
# PATHS (UNCHANGED)
# ==========================

train_dir = "U:/project/tensorf/src/abnormal_age/dataset/train"
val_dir   = "U:/project/tensorf/src/abnormal_age/dataset/validation"
MODEL_SAVE_PATH = "U:/project/tensorf/src/abnormal_age/resnet50_binary.pth"

# ==========================
# TRANSFORMS
# ==========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# DATASET (AUTO LABEL MAP)
# ==========================

train_dataset = ImageFolder(train_dir, transform=train_transform)
val_dataset   = ImageFolder(val_dir, transform=val_transform)

print("Original classes:", train_dataset.classes)

# Map labels:
# benign (0) → 0
# malignant (1) → 1
# normal (2) → 0

def map_label(label):
    return 1 if label == 1 else 0

def map_dataset(dataset):
    new_samples = []
    for path, label in dataset.samples:
        new_label = 1 if label == 1 else 0
        new_samples.append((path, new_label))
    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]

map_dataset(train_dataset)
map_dataset(val_dataset)

class_names = ['non_malignant', 'malignant']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# MODEL (RESNET50)
# ==========================

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze deeper layers
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ==========================
# LOSS & OPTIMIZER
# ==========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==========================
# TRAINING LOOP
# ==========================

best_val_acc = 0

for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ===== VALIDATION =====
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved")

# ==========================
# FINAL RESULT
# ==========================

print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")

# ==========================
# LOAD BEST MODEL
# ==========================

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False))
model.eval()

# ==========================
# PREDICTION
# ==========================

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    pred_class = class_names[predicted.item()]

    print(f"\n📷 Prediction: {pred_class}")

# ==========================
# TEST IMAGE
# ==========================

predict_image("U:/project/tensorf/src/abnormal_age/dataset/validation/benign/100_HC.png")