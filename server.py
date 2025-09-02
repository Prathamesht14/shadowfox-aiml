from flask import Flask, render_template, request, jsonify
from torchvision.models import resnet18, ResNet18_Weights
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

app = Flask(__name__, static_folder="static", template_folder=".")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

imagenet_classes = []
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

@app.route("/")
def home():
    return render_template("index.html")   

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        class_name = imagenet_classes[predicted.item()]

    return jsonify({"class_name": class_name})


if __name__ == "__main__":
    app.run(debug=True)
