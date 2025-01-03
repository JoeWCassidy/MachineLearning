import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------------------
# Task 1: Load and Visualize Data
# -------------------------------------------
def load_full_images(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    img = Image.open(part_path).convert('L')
                    img = img.resize((84, 84))
                    images.append(np.array(img))
                    labels.append(int(label_folder))
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    return np.array(images), np.array(labels)

def visualize_full_images(images, labels, num_examples=3):
    fig, axs = plt.subplots(1, num_examples, figsize=(15, 5))
    for i in range(num_examples):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------------------------
# Task 2: Baseline Models
# -------------------------------------------
def preprocess_flatten(images):
    return images.reshape(images.shape[0], -1)

def logistic_regression(x_train, y_train, x_test, y_test):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def preprocess_for_cnn(images):
    images = images / 255.0
    return images.reshape(-1, 84, 84, 1)

def create_full_image_cnn():
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

# -------------------------------------------
# Task 3: Advanced CNN Solution
# -------------------------------------------
def load_and_split_images(folder):
    images, labels = [], []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for part_file in os.listdir(label_path):
                part_path = os.path.join(label_path, part_file)
                try:
                    img = Image.open(part_path).convert('L')
                    img = img.resize((84, 84))
                    split_images = [np.array(img.crop((0, i * 28, 84, (i + 1) * 28))) for i in range(3)]
                    images.append(split_images)
                    labels.append(int(label_folder))
                except Exception as e:
                    print(f"Error loading image {part_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_split_images(images):
    images = images / 255.0
    return [images[:, i].reshape(-1, 28, 84, 1) for i in range(3)]

def preprocess_split_labels(labels, num_classes=10):
    split_labels = [[int(d) for d in f"{label:03d}"] for label in labels]
    one_hot_labels = [to_categorical([label[i] for label in split_labels], num_classes=num_classes) for i in range(3)]
    return one_hot_labels

def create_multi_output_cnn():
    input_layer = layers.Input(shape=(28, 84, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    digit1 = layers.Dense(10, activation='softmax', name='digit1')(x)
    digit2 = layers.Dense(10, activation='softmax', name='digit2')(x)
    digit3 = layers.Dense(10, activation='softmax', name='digit3')(x)
    return models.Model(inputs=input_layer, outputs=[digit1, digit2, digit3])

# -------------------------------------------
# Task 4: Model Improvement
# -------------------------------------------
def create_augmented_digit_cnn():
    model = Sequential([
        layers.Input(shape=(28, 84, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# -------------------------------------------
# Task 5: GAN for Data Augmentation
# -------------------------------------------
def build_generator(noise_dim):
    model = Sequential([
        layers.Input(shape=(noise_dim,)),
        layers.Dense(7 * 21 * 256),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 21, 256)),
        layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
        layers.Conv2D(1, (3, 3), activation='tanh', padding='same')
    ])
    return model

def build_discriminator(img_shape):
    model = Sequential([
        layers.Input(shape=img_shape),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim, save_interval):
    valid = np.ones((batch_size, 1)) * 0.9
    fake = np.zeros((batch_size, 1)) * 0.1
    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, valid)
        print(f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")

# -------------------------------------------
# Example Usage
# -------------------------------------------
if __name__ == "__main__":
    # Paths to dataset directories
    train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
    val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
    test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

    # Task 1: Load and visualize data
    print("Task 1: Loading and visualizing data...")
    train_images, train_labels = load_full_images(train_folder)
    visualize_full_images(train_images, train_labels)

    # Task 2: Baseline models
    print("\nTask 2: Training baseline models...")
    x_train_flat = preprocess_flatten(train_images)
    x_test_flat = preprocess_flatten(train_images)  # You might want to load separate test data here
    logistic_regression(x_train_flat, train_labels, x_test_flat, train_labels)

    train_images_cnn = preprocess_for_cnn(train_images)
    cnn_model = create_full_image_cnn()
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_images_cnn, train_labels, epochs=5, validation_split=0.1)
    print("\nBaseline CNN evaluation:")
    cnn_model.evaluate(train_images_cnn, train_labels, verbose=2)

    # Task 3: Advanced CNN
    print("\nTask 3: Training advanced CNNs...")
    train_images_split, train_labels_split = load_and_split_images(train_folder)
    x_train_split = preprocess_split_images(train_images_split)
    y_train_split = preprocess_split_labels(train_labels_split)
    
    advanced_cnn = create_multi_output_cnn()
    advanced_cnn.compile(optimizer='adam', 
                         loss=['categorical_crossentropy'] * 3,
                         metrics=['accuracy'])
    advanced_cnn.fit(x_train_split, 
                     {'digit1': y_train_split[0], 'digit2': y_train_split[1], 'digit3': y_train_split[2]}, 
                     epochs=5, validation_split=0.1)

    # Task 4: Model improvement
    print("\nTask 4: Improving the model...")
    improved_cnn = create_augmented_digit_cnn()
    improved_cnn.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
    improved_cnn.fit(datagen.flow(x_train_split[0], y_train_split[0], batch_size=32),
                     epochs=5, validation_split=0.1)

    # Task 5: GAN for data augmentation
    print("\nTask 5: Training GAN for data augmentation...")
    noise_dim = 100
    generator = build_generator(noise_dim)
    discriminator = build_discriminator((28, 84, 1))
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy", metrics=["accuracy"])
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")
    
    train_data = preprocess_for_cnn(train_images)
    train_dcgan(generator, discriminator, gan, train_data, epochs=100, batch_size=32, noise_dim=noise_dim, save_interval=10)
    print("GAN training completed. Synthetic data can now be generated for augmentation.")

    # Retraining best model with augmented data
    print("\nRetraining with augmented data...")
    synthetic_data = generator.predict(np.random.normal(0, 1, (1000, noise_dim)))
    augmented_data = np.vstack((train_images_cnn, synthetic_data))
    augmented_labels = np.hstack((train_labels, np.random.randint(0, 30, 1000)))
    cnn_model.fit(augmented_data, augmented_labels, epochs=5, validation_split=0.1)
    print("\nFinal model evaluation with augmented data:")
    cnn_model.evaluate(augmented_data, augmented_labels)
