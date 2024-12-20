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
from sklearn.utils.class_weight import compute_class_weight


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

# Task 2: Logistic Regression with PCA and One-vs-Rest
def preprocess_flattened_data(data_dir):
    X, y = [], []
    label_encoder = LabelEncoder()
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('L')
                X.append(np.array(img).flatten())
                y.append(os.path.basename(root))
    y = label_encoder.fit_transform(y)
    return np.array(X, dtype=np.float32), np.array(y), label_encoder

def split_labels(labels):
    return np.array([[int(label[i]) for i in range(3)] for label in labels])

def train_logistic_regression():
    X_train, y_train, label_encoder = preprocess_flattened_data(train_dir)
    X_test, y_test, _ = preprocess_flattened_data(test_dir)

    X_train /= 255.0
    X_test /= 255.0

    pca = PCA(n_components=300)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    y_train_split = split_labels(label_encoder.inverse_transform(y_train))
    y_test_split = split_labels(label_encoder.inverse_transform(y_test))

    reports = []
    for i in range(3):  # Train 3 separate models
        clf = LogisticRegression(max_iter=1000, penalty='l2')
        clf.fit(X_train_pca, y_train_split[:, i])
        y_pred = clf.predict(X_test_pca)
        report = classification_report(y_test_split[:, i], y_pred, zero_division=0)
        reports.append(f"Digit {i+1}:\n{report}\n")

    print("\n".join(reports))

# Task 2: CNN Model Fixes
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
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

def train_cnn():
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    class SubdirDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data = []
            self.labels = []
            self.transform = transform
            self.label_map = {}
            for idx, subdir in enumerate(sorted(os.listdir(data_dir))):
                self.label_map[subdir] = idx
                for file in os.listdir(os.path.join(data_dir, subdir)):
                    if file.endswith('.png'):
                        self.data.append(os.path.join(data_dir, subdir, file))
                        self.labels.append(idx)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data[idx]
            label = self.labels[idx]
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = SubdirDataset(train_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=len(train_dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

# Task 5: Improved GAN
class DCGANGenerator(nn.Module):
    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def generate_fake_images():
    generator = DCGANGenerator()
    generator.eval()
    noise = torch.randn(10, 100, 1, 1)  # Adjusted for DCGAN input format
    fake_images = generator(noise)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fake_images[i].squeeze().detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()


# Menu System
def main_menu():
    while True:
        print("\n--- Machine Learning Assignment Menu ---")
        print("1. Visualize Dataset (Task 1)")
        print("2. Train Logistic Regression (Task 2)")
        print("3. Train CNN (Task 2)")
        print("4. Generate GAN Images (Task 5)")
        print("5. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            visualize_images(train_dir)
        elif choice == "2":
            train_logistic_regression()
        elif choice == "3":
            train_cnn()
        elif choice == "4":
            generate_fake_images()
        elif choice == "5":
            print("Exiting the menu.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()

