"""
predict.py – Predict a single chest X-ray image from command line.
Usage: python predict.py path/to/image.jpg
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
model = torchvision.models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ['NORMAL', 'PNEUMONIA']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, predicted = torch.max(outputs, 1)

    pred_class = class_names[predicted.item()]
    confidence = probabilities[predicted.item()].item()
    print(f"Prediction: {pred_class} ({confidence:.2f}% confidence)")
    return pred_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    predict_image(image_path)