import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set dataset paths
train_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train"
valid_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val"
test_dir = r"C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test"

# Load Triple MNIST dataset using ImageDataGenerator
def load_triple_mnist():
    print("Loading Triple MNIST dataset...")

    # Initialize ImageDataGenerator for preprocessing
    datagen = ImageDataGenerator(rescale=1./255)

    # Load train, validation, and test sets
    train_gen = datagen.flow_from_directory(train_dir, target_size=(28, 28), color_mode='grayscale', batch_size=32, class_mode='sparse')
    valid_gen = datagen.flow_from_directory(valid_dir, target_size=(28, 28), color_mode='grayscale', batch_size=32, class_mode='sparse')
    test_gen = datagen.flow_from_directory(test_dir, target_size=(28, 28), color_mode='grayscale', batch_size=32, class_mode='sparse')

    return train_gen, valid_gen, test_gen

# Plot training curves
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.show()

# Train Decision Tree Classifier
def decision_tree_classifier(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Training Decision Tree...")
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    clf = DecisionTreeClassifier()
    clf.fit(x_train_flat, y_train)
    y_val_pred = clf.predict(x_val_flat)
    y_test_pred = clf.predict(x_test_flat)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Decision Tree Validation F1 Score: {val_f1}, Test F1 Score: {test_f1}")

# Train Basic CNN
def basic_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Training Basic CNN...")
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    model = Sequential([ 
        Dense(128, activation='relu', input_shape=(x_train_flat.shape[1],)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_flat, y_train, validation_data=(x_val_flat, y_val), epochs=20)
    test_loss, test_acc = model.evaluate(x_test_flat, y_test)
    y_test_pred = np.argmax(model.predict(x_test_flat), axis=1)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Basic CNN Test Accuracy: {test_acc}, Test F1 Score: {test_f1}")
    plot_training_curves(history)

# Train Slightly Developed CNN
def developed_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Training Slightly Developed CNN...")
    inputs = Input(shape=x_train.shape[1:])
    branch_outputs = []
    for _ in range(3):
        branch = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        branch = MaxPooling2D(pool_size=(2, 2))(branch)
        branch = Flatten()(branch)
        branch_outputs.append(branch)
    
    concatenated = layers.concatenate(branch_outputs)
    output = Dense(128, activation='relu')(concatenated)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Slightly Developed CNN Test Accuracy: {test_acc}, Test F1 Score: {test_f1}")
    plot_training_curves(history)

# Train Final CNN
def final_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    print("Training Final CNN...")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(x_train, y_train), validation_data=(x_val, y_val), epochs=25)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Final CNN Test Accuracy: {test_acc}, Test F1 Score: {test_f1}")
    plot_training_curves(history)

# Build and Train GAN
def build_and_train_gan(latent_dim, input_shape, x_train, epochs=10000, batch_size=64):
    def build_generator(latent_dim):
        model = models.Sequential()
        model.add(layers.Dense(7 * 7 * 256, input_dim=latent_dim))
        model.add(layers.Reshape((7, 7, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
        return model

    def build_discriminator(input_shape):
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=input_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def plot_generated_images(epoch, generator, latent_dim, examples=10, dim=(1, 10), figsize=(10, 1)):
        noise = np.random.randn(examples, latent_dim)
        generated_images = generator.predict(noise)
        plt.figure(figsize=figsize)
        for i in range(examples):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i, :, :, 0], interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()

    def augment_data(original_data, synthetic_data):
        return np.concatenate((original_data, synthetic_data), axis=0)

    # Build models
    discriminator = build_discriminator(input_shape)
    generator = build_generator(latent_dim)

    # Compile models
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    # Build GAN model
    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    valid = discriminator(img)
    gan = Model(z, valid)
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # Train the GAN
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]
        fake_images = generator.predict(np.random.randn(half_batch, latent_dim))
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan.train_on_batch(np.random.randn(batch_size, latent_dim), np.ones((batch_size, 1)))

        # If at save interval, save generated image samples
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            plot_generated_images(epoch, generator, latent_dim)

    return generator

# Main Menu
def main_menu():
    print("1. Train Decision Tree Classifier")
    print("2. Train Basic CNN")
    print("3. Train Slightly Developed CNN")
    print("4. Train Final CNN")
    print("5. Train GAN")
    choice = int(input("Choose an option: "))
    train_gen, valid_gen, test_gen = load_triple_mnist()

    if choice == 1:
        decision_tree_classifier(train_gen, valid_gen, test_gen)
    elif choice == 2:
        basic_cnn(train_gen, valid_gen, test_gen)
    elif choice == 3:
        developed_cnn(train_gen, valid_gen, test_gen)
    elif choice == 4:
        final_cnn(train_gen, valid_gen, test_gen)
    elif choice == 5:
        latent_dim = 100
        input_shape = train_gen.shape[1:]
        build_and_train_gan(latent_dim, input_shape, train_gen)

main_menu()
