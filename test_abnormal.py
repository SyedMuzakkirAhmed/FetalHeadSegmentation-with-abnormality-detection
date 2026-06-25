import os
import torch
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# SETTINGS
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_PATH = "U:/project/tensorf/src/abnormal_age/resnet50_binary.pth"
TEST_FOLDER = "U:/project/tensorf/src/abnormal_age/dataset/validation"
OUTPUT_FILE = "U:/project/tensorf/src/abnormal_age/test_results_binary.txt"

# Binary classes
class_names = ['non_malignant', 'malignant']

# ==========================
# LOAD MODEL
# ==========================

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# ==========================
# TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# LABEL MAPPING (CRITICAL)
# ==========================

def map_label(folder_name):
    return 1 if folder_name == "malignant" else 0

# ==========================
# TEST LOOP
# ==========================

results = []
correct = 0
total = 0

y_true = []
y_pred = []

print("\nChecking test folder:", TEST_FOLDER)

for folder in os.listdir(TEST_FOLDER):

    folder_path = os.path.join(TEST_FOLDER, folder)

    if not os.path.isdir(folder_path):
        continue

    print("\nChecking folder:", folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        pred_label = predicted.item()
        true_label = map_label(folder)

        pred_class = class_names[pred_label]
        true_class = class_names[true_label]

        # accuracy
        if pred_label == true_label:
            correct += 1

        total += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        status = "✔" if pred_label == true_label else "❌"

        print(f"{img_name} | True: {true_class} | Pred: {pred_class} {status}")

        results.append(f"{img_name},True:{true_class},Pred:{pred_class}")

# ==========================
# SAVE RESULTS
# ==========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("\n✅ Results saved to:", OUTPUT_FILE)

# ==========================
# METRICS
# ==========================

accuracy = (correct / total) * 100
print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Binary Confusion Matrix")
plt.show()

# ==========================
# REPORT
# ==========================

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))