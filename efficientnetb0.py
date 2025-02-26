import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Force the use of CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Define dataset paths
base_dir = "C:\\Users\\ASTRA\\Desktop\\Dataset"
train_dir = os.path.join(base_dir, "Training")
val_dir = os.path.join(base_dir, "Validation")
test_dir = os.path.join(base_dir, "Testing")

# Image preprocessing & augmentation
IMG_SIZE = 224
BATCH_SIZE = 32

transform = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=transform["val"])
test_dataset = datasets.ImageFolder(test_dir, transform=transform["val"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

# Check class distribution
def check_class_distribution(dataset, name):
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset:
        class_counts[dataset.classes[label]] += 1
    print(f"\n{name} Data Distribution:")
    for class_name, count in class_counts.items():
        print(f"Class '{class_name}': {count} images")

check_class_distribution(train_dataset, "Training")
check_class_distribution(val_dataset, "Validation")
check_class_distribution(test_dataset, "Testing")

# Load EfficientNetB0 model (Pretrained on ImageNet)
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features

# Modify the classifier for binary classification
model.classifier = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()  # Binary classification (Fake vs Real)
)

# Move model to CPU
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            labels = labels.float().unsqueeze(1)  # Convert labels to float for BCE loss

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total

        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%")

    return train_loss, val_loss, train_acc, val_acc

# Train the model
EPOCHS = 3
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Plot accuracy and loss
epochs_range = range(EPOCHS)

plt.plot(epochs_range, train_acc, 'r', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs_range, train_loss, 'r', label='Training Loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate on test data
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Image prediction function
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform_pipeline = transform["val"]
    img_tensor = transform_pipeline(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = "Real" if output.item() > 0.5 else "Fake"
        confidence = output.item() * 100 if output.item() > 0.5 else (1 - output.item()) * 100

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.show()

# Example prediction
img_path = "C:\\Users\\ASTRA\\Desktop\\ML\\myproject\\myapp\\image_1.png"
predict_image(img_path)

# Save model
model_save_path = "C:\\Users\\ASTRA\\Desktop\\ML\\myproject\\myapp\\efficientnet_fake_currency_model.pth"
torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
print(f"Model saved to: {model_save_path}")
