import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def logistic_regression(x_train_flattened, y_train, x_test_flattened, y_test):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train_flattened, y_train)

    y_test_pred = log_reg_model.predict(x_test_flattened)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy: {test_acc:.2f}')
    print(classification_report(y_test, y_test_pred))


def load_full_images(folder):
    images = []
    labels = []
    print(f"Scanning folder: {folder}")
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            print(f"Found label folder: {label_folder}")
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    # Load the full image (no splitting)
                    img = Image.open(part_path).convert('L')  # Convert image to grayscale
                    img = img.resize((84, 84))  # Ensure the size matches the dataset specs
                    images.append(np.array(img))
                    labels.append(label_folder)  # Use the folder name as the label
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    print(f"Loaded {len(images)} images with labels.")
    return np.array(images), np.array(labels)

# Load the dataset
test_images, test_labels = load_full_images(test_folder)

# Visualize some examples
def visualize_full_images(images, labels, num_examples=3):
    fig, axs = plt.subplots(1, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# Visualize loaded images
visualize_full_images(test_images, test_labels)
