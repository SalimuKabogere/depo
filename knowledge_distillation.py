import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# load the dataset
# Define dataset paths
DATASET_PATH = "C:\\Users\\ASTRA\\Desktop\\Dataset"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
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


# load pretrained vit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ViT
vit_teacher = models.vit_b_16(pretrained=True).to(device)

# Modify classifier head for 2 classes (Real, Fake)
num_features = vit_teacher.heads.head.in_features
vit_teacher.heads.head = nn.Linear(num_features, 2).to(device)

# Freeze ViT parameters (No training for Teacher model)
for param in vit_teacher.parameters():
    param.requires_grad = False

print("Teacher Model (ViT) Loaded")


# define CNN model

class CNNStudent(nn.Module):
    def __init__(self):
        super(CNNStudent, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Student Model
cnn_student = CNNStudent().to(device)
print("Student Model (CNN) Initialized")

# define knowledge distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Compute Soft Targets (Teacher)
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_targets = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL Divergence Loss (Soft Targets)
        loss_kl = self.criterion_kl(soft_targets, teacher_targets)

        # Cross-Entropy Loss (Hard Targets)
        loss_ce = self.criterion_ce(student_logits, labels)

        # Combined Loss
        loss = (1 - self.alpha) * loss_ce + (self.alpha * self.temperature * self.temperature) * loss_kl
        return loss


# train the student model using KD

def train_student(model, teacher, train_loader, optimizer, loss_fn, epochs=5):
    model.train()
    teacher.eval()  # Freeze teacher
    train_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Get Teacher Predictions
            with torch.no_grad():
                teacher_outputs = teacher(images)

            # Get Student Predictions
            student_outputs = model(images)

            # Compute KD Loss
            loss = loss_fn(student_outputs, teacher_outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return train_loss

# Train and evaluate

# Define Optimizer
optimizer = optim.Adam(cnn_student.parameters(), lr=0.001)
loss_fn = DistillationLoss(temperature=3.0, alpha=0.5)

# Train Student Model
EPOCHS = 5
train_loss = train_student(cnn_student, vit_teacher, train_loader, optimizer, loss_fn, EPOCHS)

# Plot Loss Curve
plt.plot(range(EPOCHS), train_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Student Model Training Loss")
plt.legend()
plt.show()


# evaluate on test data

def evaluate(model, test_loader):
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

# Evaluate Student Model
evaluate(cnn_student, test_loader)

# # Save model
# model_save_path = "C:\\Users\\ASTRA\\Desktop\\ML\\myproject\\myapp\\kd_fake_currency_model.pth"
# torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
# print(f"Model saved to: {model_save_path}")
