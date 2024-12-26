import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Conv2DTranspose, LeakyReLU
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.optimizers import Adam

# Task 1: Generate Triple-MNIST Dataset from MNIST

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize MNIST images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to create Triple-MNIST by combining three random digits
def create_triple_mnist(images, labels, num_samples=10000):
    new_images = []
    new_labels = []
    for _ in range(num_samples):
        idx = np.random.choice(len(images), 3, replace=False)
        triple_image = np.hstack((images[idx[0]], images[idx[1]], images[idx[2]]))  # Horizontally stack images
        triple_label = f"{labels[idx[0]]}{labels[idx[1]]}{labels[idx[2]]}"  # Concatenate labels as strings
        new_images.append(triple_image)
        new_labels.append(triple_label)
    return np.array(new_images), np.array(new_labels)

# Create Triple-MNIST dataset
triple_train_images, triple_train_labels = create_triple_mnist(x_train, y_train, num_samples=5000)
triple_test_images, triple_test_labels = create_triple_mnist(x_test, y_test, num_samples=1000)

# Split into train and validation sets
triple_train_images, triple_val_images, triple_train_labels, triple_val_labels = train_test_split(
    triple_train_images, triple_train_labels, test_size=0.2, random_state=42
)

# Visualize some Triple-MNIST samples
def visualize_triple_mnist(images, labels, num_samples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

visualize_triple_mnist(triple_train_images, triple_train_labels)

# Preprocessing: Reshape and normalize images for CNN
triple_train_images = triple_train_images.reshape(-1, 84, 84, 1)
triple_val_images = triple_val_images.reshape(-1, 84, 84, 1)
triple_test_images = triple_test_images.reshape(-1, 84, 84, 1)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(triple_train_labels)
y_val_encoded = label_encoder.transform(triple_val_labels)
y_test_encoded = label_encoder.transform(triple_test_labels)

# Convert to one-hot encoding for CNN
y_train_cnn = tf.keras.utils.to_categorical(y_train_encoded)
y_val_cnn = tf.keras.utils.to_categorical(y_val_encoded)
y_test_cnn = tf.keras.utils.to_categorical(y_test_encoded)

# Task 2: Baseline Logistic Regression
X_flat = triple_train_images.reshape(len(triple_train_images), -1)
X_val_flat = triple_val_images.reshape(len(triple_val_images), -1)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_flat, y_train_encoded)
y_pred = lr_model.predict(X_val_flat)

print("Logistic Regression Accuracy:", accuracy_score(y_val_encoded, y_pred))
print("Logistic Regression F1 Score:", f1_score(y_val_encoded, y_pred, average='weighted'))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_val_encoded, y_pred))

# Task 2: Basic CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Adjust for number of classes
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(triple_train_images, y_train_cnn, epochs=10, validation_data=(triple_val_images, y_val_cnn))

# Task 4: Advanced Techniques (Data Augmentation)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(triple_train_images)

advanced_cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

advanced_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
advanced_cnn_model.fit(datagen.flow(triple_train_images, y_train_cnn, batch_size=32),
                       epochs=10, validation_data=(triple_val_images, y_val_cnn))

# Task 5: Generative Adversarial Network (GAN)

# GAN Parameters
latent_dim = 100

# Generator
def build_generator():
    model = Sequential([
        Dense(21 * 21 * 128, input_dim=latent_dim),
        Reshape((21, 21, 128)),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(84, 84, 1)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# GAN Model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')

# Training GAN
def train_gan(generator, discriminator, gan, epochs=5000, batch_size=64):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, triple_train_images.shape[0], batch_size)
        real_images = triple_train_images[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss Real {d_loss_real}, D Loss Fake {d_loss_fake}, G Loss {g_loss}")

train_gan(generator, discriminator, gan)
