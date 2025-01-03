import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Function to load full images
def load_full_images(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    img = Image.open(part_path).convert('L')  # Convert image to grayscale
                    img = img.resize((84, 84))  # Resize to 84x84
                    images.append(np.array(img))
                    labels.append(int(label_folder))  # Use folder name as the label
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    return np.array(images), np.array(labels)

# Load datasets
train_images, train_labels = load_full_images(train_folder)
val_images, val_labels = load_full_images(val_folder)
test_images, test_labels = load_full_images(test_folder)

# Preprocessing for CNN
def preprocess_for_cnn(images):
    """
    Normalize pixel values and reshape to (84, 84, 1).
    """
    images = images / 255.0  # Normalize to range [0, 1]
    return images.reshape(-1, 84, 84, 1)

# Preprocess datasets
x_train_cnn = preprocess_for_cnn(train_images)
x_val_cnn = preprocess_for_cnn(val_images)
x_test_cnn = preprocess_for_cnn(test_images)

# Function to split labels into three digits
def split_labels(labels):
    """
    Split each three-digit label into a list of three individual digits.
    """
    return np.array([[int(str(label).zfill(3)[0]), int(str(label).zfill(3)[1]), int(str(label).zfill(3)[2])] 
                     for label in labels])

# Split labels into three parts
train_labels_split = split_labels(train_labels)
val_labels_split = split_labels(val_labels)
test_labels_split = split_labels(test_labels)

# Define CNN model for multi-output
def create_multi_output_cnn_model():
    base_input = layers.Input(shape=(84, 84, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(base_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Three outputs for the three digits
    digit1_output = layers.Dense(10, activation='softmax', name='digit1')(x)
    digit2_output = layers.Dense(10, activation='softmax', name='digit2')(x)
    digit3_output = layers.Dense(10, activation='softmax', name='digit3')(x)

    model = models.Model(inputs=base_input, outputs=[digit1_output, digit2_output, digit3_output])
    return model

# Train and evaluate the updated CNN
def train_and_evaluate_multi_output_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(
        optimizer='adam', 
        loss={
            'digit1': 'sparse_categorical_crossentropy', 
            'digit2': 'sparse_categorical_crossentropy', 
            'digit3': 'sparse_categorical_crossentropy'
        },
        metrics={
            'digit1': ['accuracy'],
            'digit2': ['accuracy'],
            'digit3': ['accuracy']
        }
    )
    history = model.fit(
        x_train, 
        {'digit1': y_train[:, 0], 'digit2': y_train[:, 1], 'digit3': y_train[:, 2]},
        epochs=10, 
        validation_data=(
            x_val, 
            {'digit1': y_val[:, 0], 'digit2': y_val[:, 1], 'digit3': y_val[:, 2]}
        )
    )
    test_loss = model.evaluate(
        x_test, 
        {'digit1': y_test[:, 0], 'digit2': y_test[:, 1], 'digit3': y_test[:, 2]}, 
        verbose=2
    )
    print(f'Test loss: {test_loss}')

    # Generate predictions for the test set
    y_pred = model.predict(x_test)
    y_pred_classes = [np.argmax(pred, axis=1) for pred in y_pred]

    # Print classification reports for each digit
    for i in range(3):
        print(f"Classification report for digit {i+1}:")
        print(classification_report(y_test[:, i], y_pred_classes[i]))
    
    return history

# Create and train the multi-output CNN
multi_output_cnn = create_multi_output_cnn_model()
history = train_and_evaluate_multi_output_cnn(multi_output_cnn, x_train_cnn, train_labels_split, 
                                              x_val_cnn, val_labels_split, x_test_cnn, test_labels_split)
