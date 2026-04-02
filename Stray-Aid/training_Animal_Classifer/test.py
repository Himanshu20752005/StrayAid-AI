import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("animal_model.pth"))
model.eval()

classes = ['cat', 'cow', 'dog']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
img = Image.open(r"C:\Users\Shailendra\Desktop\M_tech_projects\Strey_aid\Stray-Aid\training\Test img\nw.jpg")

img = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

print("Prediction:", classes[predicted.item()])