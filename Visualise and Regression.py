from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load dataset
def load_full_images(folder):
    images = []
    labels = []
    print(f"Scanning folder: {folder}")
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    # Load and preprocess image
                    img = Image.open(part_path).convert('L')  # Grayscale
                    img = img.resize((84, 84))  # Resize
                    images.append(np.array(img).flatten())  # Flatten image
                    labels.append(int(label_folder))  # Use folder name as label
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    print(f"Loaded {len(images)} images with labels.")
    return np.array(images), np.array(labels)

# Logistic Regression with F1 Score
from sklearn.metrics import f1_score, accuracy_score, classification_report

def logistic_regression(x_train_flattened, y_train, x_test_flattened, y_test):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train_flattened, y_train)

    y_test_pred = log_reg_model.predict(x_test_flattened)

    # Calculate accuracy
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.2f}")

    # Track which labels are problematic
    predicted_labels = set(y_test_pred)
    true_labels = set(y_test)
    missing_predictions = true_labels - predicted_labels
    missing_ground_truth = predicted_labels - true_labels

    # Display insights for undefined metrics
    if missing_predictions:
        print(f"Labels not predicted by the model: {missing_predictions}")
    if missing_ground_truth:
        print(f"Labels present in predictions but not in test set: {missing_ground_truth}")

    # Calculate F1 Score
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=1)
    print(f"Weighted F1 Score: {f1:.2f}")

    # Classification Report
    print(classification_report(y_test, y_test_pred, zero_division=1))

# Visualize dataset
def visualize_full_images(images, labels, num_examples=3):
    fig, axs = plt.subplots(1, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        axs[i].imshow(images[i].reshape(84, 84), cmap='gray')  # Reshape for visualization
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# Load datasets
train_images, train_labels = load_full_images(train_folder)
test_images, test_labels = load_full_images(test_folder)
logistic_regression(train_images, train_labels, test_images, test_labels)

# Visualize loaded images
visualize_full_images(test_images, test_labels)

# Logistic Regression Training and Evaluation
logistic_regression(train_images, train_labels, test_images, test_labels)
