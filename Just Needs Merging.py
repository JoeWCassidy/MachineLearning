###Second CNN
##import os
##import numpy as np
##from PIL import Image
##import tensorflow as tf
##from tensorflow.keras import layers, models
##from sklearn.metrics import classification_report
##import matplotlib.pyplot as plt
##
### Directories for datasets
##train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
##val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
##test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'
##
### Function to load full images
##def load_full_images(folder):
##    images = []
##    labels = []
##    for label_folder in os.listdir(folder):
##        label_path = os.path.join(folder, label_folder)
##        if os.path.isdir(label_path):
##            for part_file in os.listdir(label_path):
##                part_path = os.path.join(label_path, part_file)
##                try:
##                    img = Image.open(part_path).convert('L')  # Convert image to grayscale
##                    img = img.resize((84, 84))  # Resize to 84x84
##                    images.append(np.array(img))
##                    labels.append(int(label_folder))  # Use folder name as the label
##                except Exception as e:
##                    print(f"Error loading image {part_path}: {e}")
##    return np.array(images), np.array(labels)
##
### Load datasets
##train_images, train_labels = load_full_images(train_folder)
##val_images, val_labels = load_full_images(val_folder)
##test_images, test_labels = load_full_images(test_folder)
##
### Preprocessing for CNN
##def preprocess_for_cnn(images):
##    """
##    Normalize pixel values and reshape to (84, 84, 1).
##    """
##    images = images / 255.0  # Normalize to range [0, 1]
##    return images.reshape(-1, 84, 84, 1)
##
### Preprocess datasets
##x_train_cnn = preprocess_for_cnn(train_images)
##x_val_cnn = preprocess_for_cnn(val_images)
##x_test_cnn = preprocess_for_cnn(test_images)
##
### Function to split labels into three digits
##def split_labels(labels):
##    """
##    Split each three-digit label into a list of three individual digits.
##    """
##    return np.array([[int(str(label).zfill(3)[0]), int(str(label).zfill(3)[1]), int(str(label).zfill(3)[2])] 
##                     for label in labels])
##
### Split labels into three parts
##train_labels_split = split_labels(train_labels)
##val_labels_split = split_labels(val_labels)
##test_labels_split = split_labels(test_labels)
##
### Define CNN model for multi-output
##def create_multi_output_cnn_model():
##    base_input = layers.Input(shape=(84, 84, 1))
##    x = layers.Conv2D(32, (3, 3), activation='relu')(base_input)
##    x = layers.MaxPooling2D((2, 2))(x)
##    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
##    x = layers.MaxPooling2D((2, 2))(x)
##    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
##    x = layers.Flatten()(x)
##    x = layers.Dense(128, activation='relu')(x)
##
##    # Three outputs for the three digits
##    digit1_output = layers.Dense(10, activation='softmax', name='digit1')(x)
##    digit2_output = layers.Dense(10, activation='softmax', name='digit2')(x)
##    digit3_output = layers.Dense(10, activation='softmax', name='digit3')(x)
##
##    model = models.Model(inputs=base_input, outputs=[digit1_output, digit2_output, digit3_output])
##    return model
##
### Train and evaluate the updated CNN
##def train_and_evaluate_multi_output_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
##    model.compile(optimizer='adam', 
##                  loss={'digit1': 'sparse_categorical_crossentropy', 
##                        'digit2': 'sparse_categorical_crossentropy', 
##                        'digit3': 'sparse_categorical_crossentropy'},
##                  metrics=['accuracy'])
##    history = model.fit(x_train, {'digit1': y_train[:, 0], 'digit2': y_train[:, 1], 'digit3': y_train[:, 2]},
##                        epochs=10, 
##                        validation_data=(x_val, 
##                                         {'digit1': y_val[:, 0], 'digit2': y_val[:, 1], 'digit3': y_val[:, 2]}))
##    test_loss = model.evaluate(x_test, 
##                               {'digit1': y_test[:, 0], 'digit2': y_test[:, 1], 'digit3': y_test[:, 2]}, 
##                               verbose=2)
##    print(f'Test loss: {test_loss}')
##
##    # Generate predictions for the test set
##    y_pred = model.predict(x_test)
##    y_pred_classes = [np.argmax(pred, axis=1) for pred in y_pred]
##
##    # Print classification reports for each digit
##    for i in range(3):
##        print(f"Classification report for digit {i+1}:")
##        print(classification_report(y_test[:, i], y_pred_classes[i]))
##    
##    return history
##
### Create and train the multi-output CNN
##multi_output_cnn = create_multi_output_cnn_model()
##history = train_and_evaluate_multi_output_cnn(multi_output_cnn, x_train_cnn, train_labels_split, 
##                                              x_val_cnn, val_labels_split, x_test_cnn, test_labels_split)
###DCGAN
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Generator Model
def build_generator(noise_dim):
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(7 * 21 * 256),
        LeakyReLU(negative_slope=0.2),
        Reshape((7, 21, 256)),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2D(1, kernel_size=3, activation="tanh", padding="same")
    ])
    return model

# Discriminator Model
def build_discriminator(img_shape):
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Flatten(),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(1, activation="sigmoid")
    ])
    return model

# Assemble GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Save Generated Images
def save_generated_images(epoch, generator, noise_dim, examples=5, dim=(1, 5), figsize=(15, 2)):
    noise = np.random.normal(0, 1, (examples, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(gen_imgs[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"triple_mnist_generated_epoch_{epoch}.png")
    plt.close()

# Training Function
def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim, save_interval):
    valid = np.ones((batch_size, 1)) * 0.9  # Label smoothing
    fake = np.zeros((batch_size, 1)) * 0.1

    for epoch in range(epochs):
        # Train Discriminator
        discriminator.trainable = True
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Safely handle printing losses
        d_loss_val = d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss
        g_loss_val = g_loss[0] if isinstance(g_loss, (list, np.ndarray)) else g_loss

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss_val:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss_val:.4f}]")

        # Save generated images at intervals
        if (epoch + 1) % save_interval == 0:
            save_generated_images(epoch + 1, generator, noise_dim)

# Load and preprocess Triple MNIST data
def load_triple_mnist():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

    # Create Triple MNIST data by horizontally stacking 3 random images
    triple_data = []
    for _ in range(len(x_train)):
        digits = np.random.choice(len(x_train), 3, replace=False)
        triple_img = np.hstack([x_train[digits[0]], x_train[digits[1]], x_train[digits[2]]])
        triple_data.append(triple_img)

    triple_data = np.expand_dims(np.array(triple_data), axis=-1)
    return triple_data

# Hyperparameters
img_shape = (28, 84, 1)  # Adjusted for 28x84 combined images
noise_dim = 100
epochs = 5000
batch_size = 32
save_interval = 500

# Build and compile models
generator = build_generator(noise_dim)
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy", metrics=["accuracy"])

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

# Load data and train the DCGAN
train_data = load_triple_mnist()
train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim, save_interval)

 
#This is my final CNN 
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



#This is my second CNN 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
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
    
    # Shared feature extraction layers (no regularization and dropout)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    # Three separate branches for each digit
    digit1 = layers.Dense(256, activation='relu')(x)
    digit1 = layers.Dense(10, activation='softmax', name='digit1')(digit1)

    digit2 = layers.Dense(256, activation='relu')(x)
    digit2 = layers.Dense(10, activation='softmax', name='digit2')(digit2)

    digit3 = layers.Dense(256, activation='relu')(x)
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


#This is the rest of my code 
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load full images and labels
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

# Flatten images for Logistic Regression
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

# Preprocess images for CNN
def preprocess_for_cnn(images):
    images = images / 255.0  # Normalize to range [0, 1]
    return images.reshape(-1, 84, 84, 1)

# Preprocess labels for one-hot encoding
def preprocess_labels(labels, num_classes=10):
    one_hot_labels = []
    for label in labels:
        digits = [int(d) for d in f"{label:03d}"]
        one_hot = to_categorical(digits, num_classes=num_classes)
        one_hot_labels.append(one_hot.flatten())
    return np.array(one_hot_labels)

# Train Logistic Regression
def train_logistic_regression(x_train, y_train, x_test, y_test):
    print("Training Logistic Regression...")
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train, y_train)

    # Test Logistic Regression
    y_test_pred = log_reg_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.2f}")
    print(classification_report(y_test, y_test_pred))
    return log_reg_model

# Full Image CNN model
def create_full_image_cnn():
    model = models.Sequential()
    model.add(layers.Input(shape=(84, 84, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(30, activation='softmax'))  # Output for 3 digits * 10 classes
    return model

# Train and evaluate Full Image CNN
def train_and_evaluate_full_image_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.2f}")
    return history

# Menu
def menu():
    print("Select a task to perform:")
    print("1. Train Logistic Regression")
    print("2. Train Full Image CNN")
    print("3. Train Multi-Output CNN (Image Splits)")
    choice = input("Enter your choice: ")

    if choice == '1':
        print("Running Logistic Regression...")
        x_train_flat = flatten_images(train_images)
        x_test_flat = flatten_images(test_images)
        train_logistic_regression(x_train_flat, train_labels, x_test_flat, test_labels)
    
    elif choice == '2':
        print("Running Full Image CNN...")
        x_train_cnn = preprocess_for_cnn(train_images)
        x_val_cnn = preprocess_for_cnn(val_images)
        x_test_cnn = preprocess_for_cnn(test_images)
        train_labels_one_hot = preprocess_labels(train_labels)
        val_labels_one_hot = preprocess_labels(val_labels)
        test_labels_one_hot = preprocess_labels(test_labels)
        full_image_cnn = create_full_image_cnn()
        train_and_evaluate_full_image_cnn(full_image_cnn, x_train_cnn, train_labels_one_hot, 
                                          x_val_cnn, val_labels_one_hot, x_test_cnn, test_labels_one_hot)
    
    elif choice == '3':
        print("Running Multi-Output CNN (Image Splits)...")
        # Add logic for Multi-Output CNN with splits
        print("Implementation pending for Multi-Output CNN.")

    else:
        print("Invalid choice. Please select 1, 2, or 3.")

# Load datasets
train_images, train_labels = load_full_images(train_folder)
val_images, val_labels = load_full_images(val_folder)
test_images, test_labels = load_full_images(test_folder)

# Run menu
menu()

'''Note that the second CNN should be the second option 
the final CNN should be third option 
DCGAN should be the fourth option 
And the fifth option should be to exit 
'''
