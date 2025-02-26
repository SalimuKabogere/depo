import torch
import torchvision.models as models
import torch.nn as nn

# Load the trained model
model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features

# Modify the classifier
model.classifier = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Load trained weights
model.load_state_dict(torch.load("efficientnet_fake_currency_model.pth", map_location=torch.device("cpu")))

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")
