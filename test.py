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
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Paths to Dataset
train_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train"
valid_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val"
test_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test"

# Task 1: Visualize Dataset
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
def train_enhanced_cnn():
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
    val_dataset = SubdirDataset(valid_dir, transform)
    test_dataset = SubdirDataset(test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCNNModel(num_classes=len(train_dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    # Restore the best model weights
    early_stopping.restore_best_weights(model)

    # Test phase
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"CNN Performance:\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nConfusion Matrix:\n{cm}")

    
# Task 2: Preprocessing
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

# Task 2: Train Decision Tree with PCA
def train_decision_tree_with_pca():
    X_train, y_train, label_encoder = preprocess_flattened_data(train_dir)
    X_val, y_val, _ = preprocess_flattened_data(valid_dir)
    X_test, y_test, _ = preprocess_flattened_data(test_dir)

    # Normalize data
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    # PCA
    pca = PCA(n_components=500)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # Decision Tree with Hyperparameter Tuning
    param_grid = {
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10, 20]
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(class_weight='balanced'),
        param_grid,
        cv=3,
        scoring='accuracy'
    )
    grid_search.fit(X_train_pca, y_train)
    best_tree = grid_search.best_estimator_()

    # Evaluate
    y_pred = best_tree.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Decision Tree with PCA\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nConfusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")

# Early Stopping Implementation
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
            else:
                if self.verbose:
                    print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# Task 2: Train Enhanced CNN
class EnhancedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 10 * 10)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Task 3: Train Improved Slightly CNN
class SplitImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.label_map = {}
        for idx, subdir in enumerate(sorted(os.listdir(data_dir))):
            self.label_map[subdir] = idx
            for file in os.listdir(os.path.join(data_dir, subdir)):
                if file.endswith('.png'):
                    img_path = os.path.join(data_dir, subdir, file)
                    self.data.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')
        img_width, img_height = img.size
        left = 0
        top = 0
        right = img_width // 3
        bottom = img_height
        img_piece1 = img.crop((left, top, right, bottom))
        
        left = img_width // 3
        right = 2 * img_width // 3
        img_piece2 = img.crop((left, top, right, bottom))

        left = 2 * img_width // 3
        right = img_width
        img_piece3 = img.crop((left, top, right, bottom))

        if self.transform:
            img_piece1 = self.transform(img_piece1)
            img_piece2 = self.transform(img_piece2)
            img_piece3 = self.transform(img_piece3)

        return img_piece1, img_piece2, img_piece3, label

def train_improved_cnn():
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = SplitImageDataset(train_dir, transform)
    val_dataset = SplitImageDataset(valid_dir, transform)
    test_dataset = SplitImageDataset(test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCNNModel(num_classes=len(train_dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for img1, img2, img3, labels in train_loader:
            img1, img2, img3, labels = img1.to(device), img2.to(device), img3.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs1 = model(img1)
            outputs2 = model(img2)
            outputs3 = model(img3)
            final_outputs = torch.cat((outputs1, outputs2, outputs3), dim=1)
            loss = criterion(final_outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, img3, labels in val_loader:
                img1, img2, img3, labels = img1.to(device), img2.to(device), img3.to(device), labels.to(device)
                outputs1 = model(img1)
                outputs2 = model(img2)
                outputs3 = model(img3)
                final_outputs = torch.cat((outputs1, outputs2, outputs3), dim=1)
                loss = criterion(final_outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(final_outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    # Restore the best model weights
    early_stopping.restore_best_weights(model)

    # Test phase
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, img3, labels in test_loader:
            img1, img2, img3, labels = img1.to(device), img2.to(device), img3.to(device), labels.to(device)
            outputs1 = model(img1)
            outputs2 = model(img2)
            outputs3 = model(img3)
            final_outputs = torch.cat((outputs1, outputs2, outputs3), dim=1)
            _, preds = torch.max(final_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"CNN Performance:\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nConfusion Matrix:\n{cm}")

# Task 4: Train Improved Slightly Decision Tree
def train_improved_decision_tree():
    X_train, y_train, label_encoder = preprocess_flattened_data(train_dir)
    X_val, y_val, _ = preprocess_flattened_data(valid_dir)
    X_test, y_test, _ = preprocess_flattened_data(test_dir)

    # Normalize data
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    # PCA
    pca = PCA(n_components=500)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # Train separate trees for each part of the image
    param_grid = {
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10, 20]
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(class_weight='balanced'),
        param_grid,
        cv=3,
        scoring='accuracy'
    )

    # Train on the three pieces of the image
    # Split into 3 parts
    X_train_part1, X_train_part2, X_train_part3 = np.split(X_train_pca, 3, axis=1)
    X_val_part1, X_val_part2, X_val_part3 = np.split(X_val_pca, 3, axis=1)
    X_test_part1, X_test_part2, X_test_part3 = np.split(X_test_pca, 3, axis=1)

    # Train grid search on all parts
    grid_search.fit(X_train_part1, y_train)
    best_tree_part1 = grid_search.best_estimator_()
    grid_search.fit(X_train_part2, y_train)
    best_tree_part2 = grid_search.best_estimator_()
    grid_search.fit(X_train_part3, y_train)
    best_tree_part3 = grid_search.best_estimator_()

    # Evaluate
    y_pred_part1 = best_tree_part1.predict(X_test_part1)
    y_pred_part2 = best_tree_part2.predict(X_test_part2)
    y_pred_part3 = best_tree_part3.predict(X_test_part3)

    # Combine predictions
    final_pred = np.concatenate((y_pred_part1[:, None], y_pred_part2[:, None], y_pred_part3[:, None]), axis=1)

    # Evaluate the final prediction
    accuracy = accuracy_score(y_test, final_pred)
    f1 = f1_score(y_test, final_pred, average='weighted')
    cm = confusion_matrix(y_test, final_pred)

    print(f"Decision Tree with PCA\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\nConfusion Matrix:\n{cm}")
def check_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")
# Menu System
def main_menu():
    check_device()
    while True:
        print("\n--- Machine Learning Assignment Menu ---")
        print("1. Visualize Dataset (Task 1)")
        print("2. Train Decision Tree with PCA (Task 2)")
        print("3. Train Basic CNN (Task 2)")
        print("4. Train Improved Slightly CNN (Task 3)")
        print("5. Train Improved Slightly Decision Tree (Task 4)")
        print("6. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            visualize_images(train_dir)
        elif choice == "2":
            train_decision_tree_with_pca()
        elif choice == "3":
            train_enhanced_cnn()
        elif choice == "4":
            train_improved_cnn()
        elif choice == "5":
            train_improved_decision_tree()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
