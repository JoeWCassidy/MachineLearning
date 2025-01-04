import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, f1_score

# Load full images
def load_full_images(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(part_path, color_mode="grayscale", target_size=(84, 84))
                    images.append(np.array(img))
                    labels.append(int(label_folder))  # Use folder name as the label
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    return np.array(images), np.array(labels)

# Preprocessing for CNN
def preprocess_for_cnn(images):
    images = images / 255.0  # Normalize pixel values
    return images.reshape(-1, 84, 84, 1)  # Add channel dimension

# Split labels into three digits
def split_labels(labels):
    return np.array([[int(str(label).zfill(3)[0]), int(str(label).zfill(3)[1]), int(str(label).zfill(3)[2])] for label in labels])

# Define Final CNN model
def create_final_cnn():
    base_input = layers.Input(shape=(84, 84, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(base_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    # Three outputs for the three digits
    digit1_output = layers.Dense(10, activation='softmax', name='digit1')(x)
    digit2_output = layers.Dense(10, activation='softmax', name='digit2')(x)
    digit3_output = layers.Dense(10, activation='softmax', name='digit3')(x)

    model = models.Model(inputs=base_input, outputs=[digit1_output, digit2_output, digit3_output])
    return model

# Train and evaluate Final CNN
def train_and_evaluate_final_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
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

    # Train the model
    model.fit(
        x_train,
        {'digit1': y_train[:, 0], 'digit2': y_train[:, 1], 'digit3': y_train[:, 2]},
        validation_data=(
            x_val,
            {'digit1': y_val[:, 0], 'digit2': y_val[:, 1], 'digit3': y_val[:, 2]}
        ),
        epochs=10,
        batch_size=32
    )

    # Evaluate the model on test data
    test_loss = model.evaluate(
        x_test,
        {'digit1': y_test[:, 0], 'digit2': y_test[:, 1], 'digit3': y_test[:, 2]},
        verbose=2
    )

    print(f"Test loss: {test_loss}")

    # Predict and evaluate
    y_pred = model.predict(x_test)
    y_pred_classes = [np.argmax(pred, axis=1) for pred in y_pred]

    # Print F1 score for each digit
    for i in range(3):
        print(f"F1 score for digit {i+1}: {f1_score(y_test[:, i], y_pred_classes[i], average='macro'):.4f}")

# Paths to datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load datasets
x_train, y_train = load_full_images(train_folder)
x_val, y_val = load_full_images(val_folder)
x_test, y_test = load_full_images(test_folder)

# Preprocess datasets
x_train_cnn = preprocess_for_cnn(x_train)
x_val_cnn = preprocess_for_cnn(x_val)
x_test_cnn = preprocess_for_cnn(x_test)

# Split labels
y_train_split = split_labels(y_train)
y_val_split = split_labels(y_val)
y_test_split = split_labels(y_test)

# Create and train the model
final_cnn_model = create_final_cnn()
train_and_evaluate_final_cnn(final_cnn_model, x_train_cnn, y_train_split, x_val_cnn, y_val_split, x_test_cnn, y_test_split)
