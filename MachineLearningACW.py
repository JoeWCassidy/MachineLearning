import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Paths to Dataset
train_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train"
valid_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val"
test_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test"

# Task 1: Dataset Loading and Visualization
def visualize_images(data_dir, n_samples=5):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))
    samples = random.sample(image_paths, n_samples)
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 5))
    for i, img_path in enumerate(samples):
        img = Image.open(img_path).convert('L')
        label = os.path.basename(os.path.dirname(img_path))
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')
    plt.show()

# Dataset class for CNNs
class SubdirDataset(Dataset):
    def __init__(self, data_dir, transform=None, split_images=False):
        self.data = []
        self.labels = []
        self.transform = transform
        self.split_images = split_images
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    label = os.path.basename(root)
                    img_path = os.path.join(root, file)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')
        if self.split_images:
            img_parts = self._split_image(img)
            return img_parts, label
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def _split_image(self, img):
        width, height = img.size
        part_width = width // 3
        parts = [img.crop((i * part_width, 0, (i + 1) * part_width, height)) for i in range(3)]
        return [self.transform(part) for part in parts] if self.transform else parts

# Dataloader helpers
def get_dataloaders(data_dir, batch_size=32, split_images=False):
    transform = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
    dataset = SubdirDataset(data_dir, transform, split_images=split_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not split_images)

# Combined CNN
class CombinedCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(CombinedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 21 * 21, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 21 * 21)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Split CNN
class SplitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SplitCNN, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 21 * 21, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.shared_layers(x)
        x = x.view(-1, 64 * 21 * 21)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training helpers
def train_epoch(model, loader, device, optimizer, criterion, split_images=False):
    model.train()
    total_loss = 0
    for data in loader:
        if split_images:
            images, labels = data  # images is a list of parts, labels is a tensor
            # Move each image part to the device
            images = [img.to(device) for img in images]
        else:
            images, labels = data
            images = images.to(device)  # Move batch to the device

        # Ensure labels are moved to the device (if it's not already)
        labels = labels.to(device)  # Move the labels to the device

        optimizer.zero_grad()

        if split_images:
            # If images are split, process each part separately and combine predictions
            outputs = [model(img) for img in images]  # Get predictions for each part
            outputs = torch.cat(outputs, dim=1)  # Concatenate predictions along the feature dimension
        else:
            # For combined CNN, just forward the batch
            outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# Combined CNN training
def train_combined_cnn():
    train_loader = get_dataloaders(train_dir)
    valid_loader = get_dataloaders(valid_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        loss = train_epoch(model, train_loader, device, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        evaluate_model(model, valid_loader, device)

# Split CNN training
def train_split_cnn():
    train_loader = get_dataloaders(train_dir, split_images=True)
    valid_loader = get_dataloaders(valid_dir, split_images=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        loss = train_epoch(model, train_loader, device, optimizer, criterion, split_images=True)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        evaluate_model(model, valid_loader, device)

# Menu system
def main_menu():
    while True:
        print("\n--- Machine Learning Assignment Menu ---")
        print("1. Visualize Dataset (Task 1)")
        print("2. Train Logistic Regression (Task 2)")
        print("3. Train Combined CNN (Full Label Prediction)")
        print("4. Train Split CNN (Digit-by-Digit Prediction)")
        print("5. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            visualize_images(train_dir)
        elif choice == "2":
            print("Logistic Regression not implemented in this code snippet.")
        elif choice == "3":
            train_combined_cnn()
        elif choice == "4":
            train_split_cnn()
        elif choice == "5":
            print("Exiting the menu.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()
