import os
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("efficientnet_fake_currency_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = "Real" if output.item() > 0.5 else "Fake"
        confidence = output.item() * 100 if output.item() > 0.5 else (1 - output.item()) * 100

    return jsonify({'prediction': prediction, 'confidence': f"{confidence:.2f}%"})

# Ensure the app runs on the assigned port from Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
