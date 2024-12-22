import os
import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Paths to Dataset
train_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train"
valid_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val"
test_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test"

# Helper Function: Load Data
def load_data(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = tf.keras.utils.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)
            labels.append(int(label))
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels

# Load datasets
X_train, y_train = load_data(train_dir)
X_valid, y_valid = load_data(valid_dir)
X_test, y_test = load_data(test_dir)

# Decision Tree Model
def decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_flat, y_train)
    y_pred = dt_model.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Decision Tree - Accuracy: {acc}, F1 Score: {f1}")


# Basic CNN Model
def basic_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test):
    y_train_one_hot = to_categorical(y_train)
    y_valid_one_hot = to_categorical(y_valid)
    y_test_one_hot = to_categorical(y_test)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train_one_hot, epochs=10, validation_data=(X_valid, y_valid_one_hot))

    _, acc = model.evaluate(X_test, y_test_one_hot)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Basic CNN - Accuracy: {acc}, F1 Score: {f1}")


# Second CNN Model
def second_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test):
    def split_image(image):
        third = image.shape[1] // 3
        return [image[:, :third], image[:, third:2 * third], image[:, 2 * third:]]

    X_train_split = np.array([split_image(img) for img in X_train])
    X_valid_split = np.array([split_image(img) for img in X_valid])
    X_test_split = np.array([split_image(img) for img in X_test])

    def build_model():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        return model

    models_list = [build_model() for _ in range(3)]

    for i, model in enumerate(models_list):
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train_split[:, i], to_categorical(y_train), epochs=5, validation_data=(X_valid_split[:, i], to_categorical(y_valid)))

    # Concatenate and evaluate
    final_predictions = []
    for i, model in enumerate(models_list):
        predictions = np.argmax(model.predict(X_test_split[:, i]), axis=1)
        final_predictions.append(predictions)

    final_predictions = np.stack(final_predictions, axis=1)
    acc = accuracy_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions, average="weighted")
    print(f"Second CNN - Accuracy: {acc}, F1 Score: {f1}")


# Final CNN with Regularization and Augmentation
# Final CNN with Regularization and Data Augmentation
def final_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test):
    y_train_one_hot = to_categorical(y_train)
    y_valid_one_hot = to_categorical(y_valid)
    y_test_one_hot = to_categorical(y_test)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Model Definition
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout Layer for Regularization
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Training with Augmentation
    history = model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=64),
                        epochs=20, validation_data=(X_valid, y_valid_one_hot))

    # Evaluate the model
    _, acc = model.evaluate(X_test, y_test_one_hot)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Final CNN - Accuracy: {acc}, F1 Score: {f1}")

    # Plot Training and Validation Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()


# GAN Model
# GAN (DCGAN) Implementation
def gan_model(X_train):
    latent_dim = 100  # Size of the latent space

    # Generator Model
    def build_generator():
        model = models.Sequential([
            layers.Dense(256, activation="relu", input_dim=latent_dim),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(28 * 28, activation="tanh"),
            layers.Reshape((28, 28, 1))
        ])
        return model

    # Discriminator Model
    def build_discriminator():
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    # Compile Models
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=optimizers.Adam(0.0002), loss="binary_crossentropy", metrics=["accuracy"])

    # Combined GAN Model
    discriminator.trainable = False
    gan = models.Sequential([generator, discriminator])
    gan.compile(optimizer=optimizers.Adam(0.0002), loss="binary_crossentropy")

    # Training Loop
    epochs = 10000
    batch_size = 64
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Select Random Real Images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]

        # Generate Fake Images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)

        # Train Discriminator
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print Progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]:.2f}] [G loss: {g_loss}]")

    # Generate and Visualize Synthetic Images
    noise = np.random.normal(0, 1, (10, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]

    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()

    return generator

# Train GAN



def menu():
    while True:
        print("\nMenu:")
        print("1. Train and Evaluate Decision Tree")
        print("2. Train and Evaluate Basic CNN")
        print("3. Train and Evaluate Second CNN")
        print("4. Train and Evaluate Final CNN with Augmentation and Regularization")
        print("5. Train GAN and Generate Synthetic Images")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            print("\nRunning Decision Tree...")
            decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
        elif choice == "2":
            print("\nRunning Basic CNN...")
            basic_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test)
        elif choice == "3":
            print("\nRunning Second CNN...")
            second_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test)
        elif choice == "4":
            print("\nRunning Final CNN...")
            final_cnn(X_train, y_train, X_valid, y_valid, X_test, y_test)
        elif choice == "5":
            print("\nTraining GAN and Generating Synthetic Images...")
            gan_model(X_train)
        elif choice == "6":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

# Run the menu
menu()

