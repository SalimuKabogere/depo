import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define dataset paths
base_dir = "C:\\Users\\ASTRA\\Desktop\\Dataset"
train_dir = os.path.join(base_dir, "Training")
val_dir = os.path.join(base_dir, "Validation")
test_dir = os.path.join(base_dir, "Testing")

categories = ["Real", "Fake"]

# Function to extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys")
    
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
    
    hist_b /= np.sum(hist_b) if np.sum(hist_b) != 0 else 1
    hist_g /= np.sum(hist_g) if np.sum(hist_g) != 0 else 1
    hist_r /= np.sum(hist_r) if np.sum(hist_r) != 0 else 1
    
    features = np.hstack([hog_features, hist_b, hist_g, hist_r])
    return features

# Function to load dataset
def load_dataset(data_path):
    X, y = [], []
    for label, category in enumerate(categories):
        folder_path = os.path.join(data_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

# Load all datasets
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)
X_test, y_test = load_dataset(test_dir)

input_size = X_train.shape[1]  # Dynamically get feature size

# Custom dataset class
class CurrencyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create DataLoaders
train_dataset = CurrencyDataset(X_train, y_train)
val_dataset = CurrencyDataset(X_val, y_val)
test_dataset = CurrencyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Define neural network model
class CurrencyClassifier(nn.Module):
    def __init__(self, input_size):
        super(CurrencyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model with correct input size
model = CurrencyClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model():
    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in train_loader:
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc * 100:.2f}%")

# Train the model
train_model()

# Evaluate function
def evaluate():
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=categories))

# Evaluate the model
evaluate()

# Save model
model_save_path = "C:\\Users\\ASTRA\\Desktop\\ML\\myproject\\myapp\\random_fake_currency_model.pth"
torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
print(f"Model saved to: {model_save_path}")
