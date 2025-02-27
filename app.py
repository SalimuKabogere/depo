from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.feature import hog
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the same neural network model as before
class CurrencyClassifier(nn.Module):
    def __init__(self, input_size):
        super(CurrencyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Use the correct input size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: Real & Fake
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax, CrossEntropyLoss handles it
        return x

# Initialize and load the model
input_size = 8868  # Update this to the correct input size
model = CurrencyClassifier(input_size)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use a relative path to the model file
model_path = os.path.join(script_dir, 'random_fake_currency_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to extract features (same as before)
def extract_features(image):
    if image is None:
        return None  # Skip invalid images

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

# Route to serve the index.html page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Read the image
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Extract features
    features = extract_features(img)
    if features is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Convert features to tensor and make prediction
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    result = "Real" if predicted.item() == 0 else "Fake"
    confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()

    print(f"Prediction: {result}, Confidence: {confidence}")

    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)