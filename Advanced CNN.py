import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load images and labels
def load_images_and_labels(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    img = Image.open(file_path).convert('L')  # Convert to grayscale
                    img = img.resize((84, 84))  # Resize to 84x84
                    images.append(np.array(img))
                    labels.append(int(label_folder))  # Use folder name as label
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)

# Preprocess images and labels
def preprocess_images_and_labels(images, labels, num_classes=10):
    images = images / 255.0  # Normalize to range [0, 1]
    images = images.reshape(-1, 84, 84, 1)  # Add channel dimension
    split_labels = [[int(d) for d in f"{label:03d}"] for label in labels]  # Split into three digits
    one_hot_labels = [to_categorical([label[i] for label in split_labels], num_classes=num_classes) for i in range(3)]
    return images, one_hot_labels

# Define a combined CNN model
def create_combined_cnn():
    inputs = layers.Input(shape=(84, 84, 1))
    
    # Shared feature extraction layers
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    # Three separate branches for each digit
    digit1 = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    digit1 = layers.Dropout(0.5)(digit1)
    digit1 = layers.Dense(10, activation='softmax', name='digit1')(digit1)

    digit2 = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    digit2 = layers.Dropout(0.5)(digit2)
    digit2 = layers.Dense(10, activation='softmax', name='digit2')(digit2)

    digit3 = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    digit3 = layers.Dropout(0.5)(digit3)
    digit3 = layers.Dense(10, activation='softmax', name='digit3')(digit3)

    model = models.Model(inputs=inputs, outputs=[digit1, digit2, digit3])
    return model

# Plot loss curves
def plot_combined_loss_curves(history):
    plt.figure(figsize=(15, 10))
    legend_labels = []  # Track labels for the legend
    for i in range(3):
        train_loss_key = f'digit{i + 1}_loss'
        val_loss_key = f'val_digit{i + 1}_loss'
        
        # Check if the keys exist in the history
        if train_loss_key in history.history:
            plt.plot(history.history[train_loss_key], label=f"Digit {i + 1} - Training Loss")
            legend_labels.append(f"Digit {i + 1} - Training Loss")
        if val_loss_key in history.history:
            plt.plot(history.history[val_loss_key], linestyle="--", label=f"Digit {i + 1} - Validation Loss")
            legend_labels.append(f"Digit {i + 1} - Validation Loss")
    
    if legend_labels:
        plt.legend()  # Create legend if any labels are added
    plt.title("Training and Validation Loss for Combined Model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# Load and preprocess datasets
train_images, train_labels = load_images_and_labels(train_folder)
val_images, val_labels = load_images_and_labels(val_folder)
test_images, test_labels = load_images_and_labels(test_folder)

x_train, y_train = preprocess_images_and_labels(train_images, train_labels)
x_val, y_val = preprocess_images_and_labels(val_images, val_labels)
x_test, y_test = preprocess_images_and_labels(test_images, test_labels)

# Create the combined CNN
model = create_combined_cnn()

# Compile the model with separate loss and metrics for each output
model.compile(
    optimizer='adam',
    loss={
        'digit1': 'categorical_crossentropy',
        'digit2': 'categorical_crossentropy',
        'digit3': 'categorical_crossentropy',
    },
    metrics={
        'digit1': ['accuracy'],
        'digit2': ['accuracy'],
        'digit3': ['accuracy'],
    }
)

# Define EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train, 
    {'digit1': y_train[0], 'digit2': y_train[1], 'digit3': y_train[2]},
    validation_data=(x_val, {'digit1': y_val[0], 'digit2': y_val[1], 'digit3': y_val[2]}),
    batch_size=32,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set
evaluation = model.evaluate(
    x_test, 
    {'digit1': y_test[0], 'digit2': y_test[1], 'digit3': y_test[2]}, 
    verbose=1
)

# Print evaluation results
print("Evaluation output:", evaluation)
if len(evaluation) >= 7:
    test_loss = evaluation[0]
    digit1_accuracy = evaluation[2]
    digit2_accuracy = evaluation[4]
    digit3_accuracy = evaluation[6]
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Digit 1 Accuracy: {digit1_accuracy:.4f}")
    print(f"Digit 2 Accuracy: {digit2_accuracy:.4f}")
    print(f"Digit 3 Accuracy: {digit3_accuracy:.4f}")
else:
    print("Unexpected evaluation output structure. Please debug further.")

# Predict on test data
predictions = model.predict(x_test)

# Concatenate predictions to form full labels
full_predictions = np.array([int("".join(map(str, pred))) for pred in zip(
    np.argmax(predictions[0], axis=1), 
    np.argmax(predictions[1], axis=1), 
    np.argmax(predictions[2], axis=1)
)])

# Evaluate full label accuracy
accuracy = accuracy_score(test_labels, full_predictions)
f1 = f1_score(test_labels, full_predictions, average="macro")

print(f"Full label accuracy: {accuracy:.2f}")
print(f"Full label F1 score: {f1:.2f}")
print("Classification Report for Full Labels:")
print(classification_report(test_labels, full_predictions))

# Plot the loss curves
plot_combined_loss_curves(history)
