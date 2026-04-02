import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Step 1: Image transformations (IMPROVED)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Step 2: Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Classes:", dataset.classes)

# Step 3: Load pretrained model
model = models.resnet18(pretrained=True)

# 🔥 Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# Step 4: Loss and optimizer (LOWER LR)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0003)

# Step 5: Training loop
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Step 6: Save model
torch.save(model.state_dict(), "animal_model.pth")

print("✅ Training complete! Model saved as animal_model.pth")