import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from PIL import Image

# --------------------- Dataset Loading and Preprocessing ---------------------

def load_image(file_path, target_size=(28, 28)):
    # Open image, resize and normalize
    img = Image.open(file_path).convert('RGB')  # Convert to RGB to ensure 3 channels
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

def split_image(img, left_width=9, center_width=10, right_width=9):
    # Split the image into three sections: left, center, right
    left = img[:, :left_width]
    center = img[:, left_width:left_width+center_width]
    right = img[:, left_width+center_width:]
    return left, center, right

def load_dataset(base_path, split="train", target_size=(28, 28)):
    images = []
    labels = []
    categories = set()  # This will store unique categories

    # Scan through the subdirectories to gather all categories
    for category_folder in os.listdir(os.path.join(base_path, split)):
        # Skip folders that don't match the expected 3-digit pattern
        if len(category_folder) != 3 or not category_folder.isdigit():
            print(f"Skipping folder: {category_folder}")
            continue
        
        category = int(category_folder)  # Convert folder name to integer label
        categories.add(category)  # Add to the set of categories
        
        label_path = os.path.join(base_path, split, category_folder)
        image_files = [f for f in os.listdir(label_path) if f.endswith('.png')]

        for image_file in image_files:
            img_path = os.path.join(label_path, image_file)
            img = load_image(img_path, target_size)

            images.append(img)
            labels.append(category)  # Use category as label

    images = np.array(images)
    labels = np.array(labels)

    # Sort categories to ensure consistent labeling
    categories = sorted(categories)
    category_map = {category: idx for idx, category in enumerate(categories)}

    # Map labels to indices based on category map
    labels = np.array([category_map[label] for label in labels])

    return images, labels, category_map

def load_dataset_with_split(base_path, split="train", target_size=(28, 28)):
    images = []
    labels = []
    categories = set()  # This will store unique categories

    for category_folder in os.listdir(os.path.join(base_path, split)):
        # Skip folders that don't match the expected 3-digit pattern
        if len(category_folder) != 3 or not category_folder.isdigit():
            print(f"Skipping folder: {category_folder}")
            continue

        category = int(category_folder)  # Convert folder name to integer label
        categories.add(category)  # Add to the set of categories
        
        label_path = os.path.join(base_path, split, category_folder)
        image_files = [f for f in os.listdir(label_path) if f.endswith('.png')]

        for image_file in image_files:
            img_path = os.path.join(label_path, image_file)
            img = load_image(img_path, target_size)
            
            # Split image into 3 parts
            left, center, right = split_image(img)

            images.append((left, center, right))
            labels.append(category)  # Use category as label

    # Convert to numpy arrays
    left_images = np.array([item[0] for item in images])
    center_images = np.array([item[1] for item in images])
    right_images = np.array([item[2] for item in images])
    labels = np.array(labels)

    # Sort categories and create a mapping
    categories = sorted(categories)
    category_map = {category: idx for idx, category in enumerate(categories)}

    # Map labels to indices based on category map
    labels = np.array([category_map[label] for label in labels])

    return left_images, center_images, right_images, labels, category_map

# Example usage:
base_path = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist"
# Load train, test, and validation datasets without manually specifying categories
X_train, y_train, category_map_train = load_dataset(base_path, split="train")
X_val, y_val, category_map_val = load_dataset(base_path, split="val")
X_test, y_test, category_map_test = load_dataset(base_path, split="test")

# Load dataset with split images
X_train_left, X_train_center, X_train_right, y_train, category_map_train_split = load_dataset_with_split(base_path, split="train")
X_val_left, X_val_center, X_val_right, y_val, category_map_val_split = load_dataset_with_split(base_path, split="val")
X_test_left, X_test_center, X_test_right, y_test, category_map_test_split = load_dataset_with_split(base_path, split="test")

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(category_map_train))
y_val = to_categorical(y_val, num_classes=len(category_map_val))
y_test = to_categorical(y_test, num_classes=len(category_map_test))

# Check the shapes of the data
print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, Labels shape: {y_val.shape}")
print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

# --------------------- Define the CNN Models ---------------------

def create_cnn_model(input_shape=(28, 28, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(category_map_train), activation='softmax'))  # Dynamically use the number of categories
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_split_cnn_model(input_shape=(28, 28, 3)):
    left_input = layers.Input(shape=input_shape)
    center_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    x1 = layers.Conv2D(32, (3, 3), activation='relu')(left_input)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.Flatten()(x1)

    x2 = layers.Conv2D(32, (3, 3), activation='relu')(center_input)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = layers.Flatten()(x2)

    x3 = layers.Conv2D(32, (3, 3), activation='relu')(right_input)
    x3 = layers.MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = layers.Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = layers.Flatten()(x3)

    concatenated = layers.concatenate([x1, x2, x3])
    x = layers.Dense(128, activation='relu')(concatenated)
    x = layers.Dense(len(category_map_train), activation='softmax')(x)  # Use the number of categories dynamically

    model = models.Model(inputs=[left_input, center_input, right_input], outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------- Training the Models ---------------------

# Create and train CNN model
cnn_model = create_cnn_model(input_shape=(28, 28, 3))
cnn_model.summary()
cnn_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val))

# Create and train split CNN model
split_cnn_model = create_split_cnn_model(input_shape=(28, 28, 3))
split_cnn_model.summary()
split_cnn_model.fit([X_train_left, X_train_center, X_train_right], y_train, epochs=10, batch_size=8, validation_data=([X_val_left, X_val_center, X_val_right], y_val))

