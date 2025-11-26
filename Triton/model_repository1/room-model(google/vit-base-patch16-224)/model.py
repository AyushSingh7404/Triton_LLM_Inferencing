import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGCustom(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGCustom, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),  # Because 300/32 ≈ 9 after 5 poolings
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

# Example usage
model = VGGCustom(num_classes=2)
print(model)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class RoomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Prepare paths and labels
base_path ="C:/Users/rajpu/Desktop/Trinton/rooms_dataset"

room_types = ["bedroom", "diningroom"]
image_paths = []
labels = []

for idx, room in enumerate(room_types):
    folder = os.path.join(base_path, room)
    for img_name in os.listdir(folder):
        image_paths.append(os.path.join(folder, img_name))
        labels.append(idx)

# Transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = RoomDataset(image_paths, labels, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = VGGCustom(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train (just for demo, add epochs, etc.)
model.train()
for epoch in range(3):
    total_loss = 0
    for imgs, lbls in loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# Save as TorchScript
model.eval()
example_input = torch.randn(1, 3, 300, 300)
traced_model = torch.jit.trace(model, example_input)

os.makedirs("model_repository/room_model/1", exist_ok=True)
traced_model.save("model_repository/room_model/1/model.pt")
print("Saved TorchScript model as model.pt")
