import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

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
            for file in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file)
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

# GAN Functions
def build_generator(noise_dim):
    """
    Create the generator model for GAN.
    """
    model = Sequential([
        layers.Input(shape=(noise_dim,)),
        layers.Dense(21 * 21 * 256),
        layers.Reshape((21, 21, 256)),
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="tanh"),
        layers.Reshape((84, 84, 1))
    ])
    return model

def build_discriminator(img_shape):
    """
    Create the discriminator model for GAN.
    """
    model = Sequential([
        layers.Input(shape=img_shape),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(128),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim):
    """
    Train the GAN model.
    """
    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

def integrate_synthetic_images(generator, x_train, num_images):
    """
    Generate synthetic images using the GAN and integrate them into the training dataset.
    """
    noise = np.random.normal(0, 1, (num_images, 100))
    synthetic_images = generator.predict(noise)
    synthetic_images = 0.5 * synthetic_images + 0.5  # Rescale to [0, 1]
    x_augmented = np.vstack((x_train, synthetic_images))
    return x_augmented

# Train and Evaluate CNNs with GAN Integration
def train_cnn_with_gan_augmentation():
    print("\nTraining CNNs with GAN-Augmented Dataset...")

    # Generate synthetic images
    noise_dim = 100
    generator = build_generator(noise_dim)
    discriminator = build_discriminator((84, 84, 1))
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    print("\nTraining GAN...")
    train_dcgan(generator, discriminator, gan, x_train, epochs=500, batch_size=32, noise_dim=noise_dim)

    print("\nGenerating synthetic images...")
    x_augmented = integrate_synthetic_images(generator, x_train, 5000)

    # Retrain Basic CNN
    print("\nRetraining Basic CNN...")
    basic_cnn = create_basic_cnn()
    basic_cnn.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    basic_cnn.fit(x_augmented, np.hstack((y_train[:, 0], y_train[:, 0][:5000])), epochs=5)
    evaluate_model(basic_cnn, x_test, y_test[:, 0], "Basic_CNN_With_GAN")

    # Retrain Advanced CNN
    print("\nRetraining Advanced CNN...")
    advanced_cnn = create_advanced_cnn()
    advanced_cnn.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy'] * 3, metrics=['accuracy'])
    advanced_cnn.fit(
        x_augmented, 
        [np.hstack((y_train[:, i], y_train[:, i][:5000])) for i in range(3)], 
        epochs=5
    )
    evaluate_model(advanced_cnn, x_test, y_test, "Advanced_CNN_With_GAN")

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

# Train and Evaluate CNNs with GAN-Augmented Dataset
train_cnn_with_gan_augmentation()
