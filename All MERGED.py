import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Utility Functions
def load_images_and_labels(folder, image_size=(84, 84)):
    images, labels = [], []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    img = Image.open(file_path).convert('L').resize(image_size)
                    images.append(np.array(img))
                    labels.append(int(label_folder))
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_images(images):
    images = images / 255.0
    return images.reshape(-1, 84, 84, 1)

def split_labels(labels):
    return np.array([[int(str(label).zfill(3)[0]), int(str(label).zfill(3)[1]), int(str(label).zfill(3)[2])] 
                     for label in labels])

def visualize_images(images, labels, num_examples=5):
    fig, axs = plt.subplots(1, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# Logistic Regression with Metrics
def logistic_regression(x_train, y_train, x_test, y_test):
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train_flat, y_train)

    y_test_pred = log_reg_model.predict(x_test_flat)
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
    print(classification_report(y_test, y_test_pred))
    f1 = f1_score(y_test, y_test_pred, average="weighted")
    print(f"F1 Score: {f1:.2f}")
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# CNN Architectures
def create_basic_cnn():
    model = Sequential([
        layers.Input(shape=(84, 84, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(30, activation='softmax')
    ])
    return model

# Data Augmentation
def augment_data(x_train):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    return datagen

# GAN Implementation
def build_generator(noise_dim):
    model = Sequential([
        layers.Dense(21 * 21 * 256, input_dim=noise_dim),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((21, 21, 256)),
        layers.BatchNormalization(momentum=0.8),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.BatchNormalization(momentum=0.8),
        layers.Conv2D(1, kernel_size=3, activation="tanh", padding="same")  # Output shape: (84, 84, 1)
    ])
    return model

def build_discriminator(img_shape):
    model = Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim):
    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_imgs = generator.predict(noise)
        discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))

# Integrate GAN-generated Images
def integrate_synthetic_images(generator, x_train, num_images):
    noise = np.random.normal(0, 1, (num_images, 100))
    synthetic_images = generator.predict(noise)
    synthetic_images = 0.5 * synthetic_images + 0.5  # Rescale to [0, 1]
    x_augmented = np.vstack((x_train, synthetic_images))
    return x_augmented

# Main Workflow
train_images, train_labels = load_images_and_labels(train_folder)
val_images, val_labels = load_images_and_labels(val_folder)
test_images, test_labels = load_images_and_labels(test_folder)
x_train = preprocess_images(train_images)
x_val = preprocess_images(val_images)
x_test = preprocess_images(test_images)
y_train = split_labels(train_labels)
y_val = split_labels(val_labels)
y_test = split_labels(test_labels)

# Data Augmentation Example
datagen = augment_data(x_train)

# GAN Example
noise_dim = 100
generator = build_generator(noise_dim)
discriminator = build_discriminator((84, 84, 1))
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(), loss='binary_crossentropy')
train_dcgan(generator, discriminator, gan, x_train, epochs=10, batch_size=32, noise_dim=noise_dim)

# Integrate Synthetic Images Example
x_augmented = integrate_synthetic_images(generator, x_train, 5000)
