import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# load dataset
# Dataset Path
DATASET_PATH = "C:\\Users\\ASTRA\\Desktop\\Dataset"

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for better training
])

# Load Dataset
BATCH_SIZE = 32

train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "Training"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "Validation"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "Testing"), transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print dataset info
print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# load pretrained vit model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ViT Model
vit_model = models.vit_b_16(pretrained=True).to(device)

# Modify Classifier for 2 Classes (Real, Fake)
num_features = vit_model.heads.head.in_features
vit_model.heads.head = nn.Linear(num_features, 2).to(device)

print("ViT Model Loaded and Modified for Fake Currency Detection")

# train vit model

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

    print("Training Completed!")

# Train ViT Model
EPOCHS = 3
train_model(vit_model, train_loader, val_loader, optimizer, criterion, EPOCHS)


# evaluate model on test
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Evaluate Trained Model
evaluate_model(vit_model, test_loader)

# save the model
torch.save(vit_model.state_dict(), "vit_fake_currency.pth")
print("Model Saved Successfully!")

