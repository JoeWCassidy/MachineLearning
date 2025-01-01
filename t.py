import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Clear any previous TensorFlow sessions
tf.keras.backend.clear_session()

# Paths to dataset directories
base_dir = "C:/Users/josep/Documents/GitHub/MachineLearning/dataset2/triple_mnist"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Data preprocessing
image_size = (64, 64)  # Resize all images to 64x64
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Function to log and verify dataset structures
def log_dataset_structure(directory):
    classes = os.listdir(directory)
    class_count = len(classes)
    print(f"Found {class_count} classes in {directory}: {classes[:5]}...")  # Show first 5 classes
    return class_count

# Log and verify class counts for all datasets
train_class_count = log_dataset_structure(train_dir)
val_class_count = log_dataset_structure(val_dir)
test_class_count = log_dataset_structure(test_dir)

# Check for class mismatch
if train_class_count != val_class_count or train_class_count != test_class_count:
    raise ValueError(
        f"Inconsistent class counts: train={train_class_count}, val={val_class_count}, test={test_class_count}. "
        "Ensure all datasets have the same classes."
    )

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Ensures one-hot encoding for labels
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Ensures one-hot encoding for labels
    color_mode='grayscale'
)

test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Ensures one-hot encoding for labels
    color_mode='grayscale'
)

# Use the number of classes from the training generator
num_classes = train_generator.num_classes
print(f"Number of classes detected: {num_classes}")

# CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Match output layer to number of classes
])

# Compile the model
cnn_model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# Summary of the model
cnn_model.summary()

# Training the CNN model
history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=10
)

# Visualization of training results
import matplotlib.pyplot as plt

def plot_training(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# Plot the training history
plot_training(history)
