import os
import tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load images and labels from folders
def load_full_images(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    img = Image.open(part_path).convert('L')  # Convert to grayscale
                    img = img.resize((84, 84))  # Resize to 84x84
                    images.append(np.array(img))
                    labels.append(int(label_folder))  # Use folder name as label
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    return np.array(images), np.array(labels)

# Normalize images and reshape for CNN
def preprocess_for_cnn(images):
    images = images / 255.0  # Normalize to range [0, 1]
    return images.reshape(-1, 84, 84, 1)

# One-hot encode labels for 3-digit prediction
def preprocess_labels(labels, num_classes=10):
    """
    Convert three-digit labels into one-hot encoded format.
    Each label is represented as a concatenation of three one-hot vectors.
    """
    one_hot_labels = []
    for label in labels:
        digits = [int(d) for d in f"{label:03d}"]  # Ensure three digits
        one_hot = to_categorical(digits, num_classes=num_classes)
        one_hot_labels.append(one_hot.flatten())
    return np.array(one_hot_labels)

# Define the CNN model
def create_full_image_cnn():
    model = models.Sequential()
    model.add(layers.Input(shape=(84, 84, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))  # Reduced dense layer size
    model.add(layers.Dense(30, activation='softmax'))  # Output for 3 digits * 10 classes
    return model

# Train and evaluate the CNN
def train_and_evaluate_full_image_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.2f}")
    
    # Predict on test set and decode results
    y_test_pred = model.predict(x_test)
    y_test_pred_digits = np.argmax(y_test_pred.reshape(-1, 3, 10), axis=-1)
    y_test_actual_digits = np.argmax(y_test.reshape(-1, 3, 10), axis=-1)
    
    # Convert predictions back to three-digit labels
    y_test_pred_labels = np.array([int("".join(map(str, digits))) for digits in y_test_pred_digits])
    y_test_actual_labels = np.array([int("".join(map(str, digits))) for digits in y_test_actual_digits])
    
    # Print classification report
    print(classification_report(y_test_actual_labels, y_test_pred_labels))
    return history

# Load the datasets
train_images, train_labels = load_full_images(train_folder)
val_images, val_labels = load_full_images(val_folder)
test_images, test_labels = load_full_images(test_folder)

# Preprocess images and labels
x_train_cnn = preprocess_for_cnn(train_images)
x_val_cnn = preprocess_for_cnn(val_images)
x_test_cnn = preprocess_for_cnn(test_images)

train_labels_one_hot = preprocess_labels(train_labels)
val_labels_one_hot = preprocess_labels(val_labels)
test_labels_one_hot = preprocess_labels(test_labels)

# Create, train, and evaluate the model
full_image_cnn = create_full_image_cnn()
history = train_and_evaluate_full_image_cnn(full_image_cnn, x_train_cnn, train_labels_one_hot, 
                                            x_val_cnn, val_labels_one_hot, x_test_cnn, test_labels_one_hot)
