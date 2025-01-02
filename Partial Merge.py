import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, MaxPooling2D, Dropout
)

# Directories for datasets
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Utility Functions
def load_images_and_labels(folder, image_size=(84, 84)):
    """Load images and labels from a folder."""
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

def preprocess_images_and_labels(images, labels, num_classes=10):
    """Normalize images and convert labels to one-hot encoding."""
    images = images / 255.0
    images = images.reshape(-1, 84, 84, 1)
    split_labels = [[int(d) for d in f"{label:03d}"] for label in labels]
    one_hot_labels = [to_categorical([label[i] for label in split_labels], num_classes=num_classes) for i in range(3)]
    return images, one_hot_labels

# Model Definitions
def create_basic_cnn():
    """Basic CNN for full image prediction."""
    model = Sequential([
        Input(shape=(84, 84, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(30, activation='softmax')  # 3 digits * 10 classes
    ])
    return model

def create_advanced_cnn():
    """Advanced CNN with separate outputs for each digit."""
    inputs = Input(shape=(84, 84, 1))
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    digit_outputs = []
    for _ in range(3):
        digit = Dense(256, activation='relu', kernel_regularizer='l2')(x)
        digit = Dropout(0.5)(digit)
        digit_outputs.append(Dense(10, activation='softmax')(digit))

    model = models.Model(inputs=inputs, outputs=digit_outputs)
    return model

def build_generator(noise_dim):
    """DCGAN Generator."""
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(7 * 21 * 256),
        LeakyReLU(0.2),
        Reshape((7, 21, 256)),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, 3, strides=2, padding="same"),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, 3, strides=2, padding="same"),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Conv2D(1, 3, activation="tanh", padding="same")
    ])
    return model

def build_discriminator(img_shape):
    """DCGAN Discriminator."""
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(64, 3, strides=2, padding="same"),
        LeakyReLU(0.2),
        Flatten(),
        Dense(128),
        LeakyReLU(0.2),
        Dense(1, activation="sigmoid")
    ])
    return model

# Train and Evaluate Models
def train_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test, is_advanced=False):
    """Train CNN model and evaluate."""
    loss = 'categorical_crossentropy' if not is_advanced else {
        'output1': 'categorical_crossentropy',
        'output2': 'categorical_crossentropy',
        'output3': 'categorical_crossentropy'
    }
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=20, callbacks=[early_stopping])

    test_loss, *test_metrics = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Metrics: {test_metrics}")

# GAN Training and Image Generation
def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim):
    """Train DCGAN."""
    valid, fake = np.ones((batch_size, 1)), np.zeros((batch_size, 1))
    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, valid)

        print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")
