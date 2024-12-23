import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Reshape,Conv2DTranspose,Conv2D,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_triple_mnist():
    # Replace this with actual loading of the Triple MNIST dataset
    print("Loading Triple MNIST dataset...")
    # Placeholder: Load standard MNIST for demonstration purposes
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=-1) / 255.0
    x_val = np.expand_dims(x_val, axis=-1) / 255.0
    x_test = np.expand_dims(x_test, axis=-1) / 255.0
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

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

def build_gan_and_train(x_train, epochs=50, batch_size=64):
    img_shape = x_train.shape[1:]
    latent_dim = 100

    # Generator
    def build_generator():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(np.prod(img_shape), activation='tanh'),
            tf.keras.layers.Reshape(img_shape)
        ])
        return model

    # Discriminator
    def build_discriminator():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=img_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    # Instantiate Generator and Discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile the Discriminator
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Build and Compile the GAN
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )

    def train_gan():
        epochs=25
        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                # Get a random batch of real images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_images = x_train[idx]
                real_labels = np.ones((batch_size, 1))

                # Generate fake images
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                fake_images = generator.predict(noise)
                fake_labels = np.zeros((batch_size, 1))

                # Log shapes of the images and labels
                print(f"Real images shape: {real_images.shape}, Real labels shape: {real_labels.shape}")
                print(f"Fake images shape: {fake_images.shape}, Fake labels shape: {fake_labels.shape}")

                # Train the discriminator
                try:
                    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
                    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
                    print(f"D Real Loss: {d_loss_real}, D Fake Loss: {d_loss_fake}")
                except Exception as e:
                    print(f"Error during Discriminator training: {e}")
                    break

                # Train the generator
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                valid_labels = np.ones((batch_size, 1))  # Labels for fake images should be 1 when training generator
                try:
                    g_loss = gan.train_on_batch(noise, valid_labels)
                    print(f"G Loss: {g_loss}")
                except Exception as e:
                    print(f"Error during Generator training: {e}")
                    break

                # Optional: Save or visualize generated images
                if epoch % save_interval == 0 or epoch == epochs - 1:
                    generate_and_save_images(generator, epoch)

        except Exception as e:
            print(f"Error in GAN training loop: {e}")




    train_gan()
    return generator

def generate_and_visualize_images(generator, num_images=10, latent_dim=100):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images * 127.5 + 127.5).astype('uint8')  # Scale back to [0, 255]

    # Visualize synthetic images
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()
    return generated_images

def augment_and_retrain(generator, x_train, y_train, x_val, y_val, x_test, y_test, num_augmented=5000, latent_dim=100):
    # Generate synthetic images
    synthetic_images = generate_and_visualize_images(generator, num_augmented, latent_dim)

    # Create synthetic labels (random labels for augmentation, can be refined if needed)
    synthetic_labels = np.random.choice(y_train, num_augmented)

    # Combine datasets
    x_train_augmented = np.concatenate([x_train, synthetic_images], axis=0)
    y_train_augmented = np.concatenate([y_train, synthetic_labels], axis=0)

    # Retrain the Final CNN
    print("Retraining Final CNN with augmented data...")
    final_cnn(x_train_augmented, y_train_augmented, x_val, y_val, x_test, y_test)

def dcgan(x_train, y_train, x_test, y_test):
    print("Training DCGAN...")
    generator = build_gan_and_train(x_train, epochs=25, batch_size=32)
    print("Generating synthetic images...")
    generate_and_visualize_images(generator, num_images=10)
    augment_and_retrain(generator, x_train, y_train, x_test, y_test, x_test, y_test)


def main_menu():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_triple_mnist()
    while True:
        print("\nMain Menu")
        print("1. Decision Tree")
        print("2. Basic CNN")
        print("3. Slightly Developed CNN")
        print("4. Final CNN")
        print("5. DCGAN")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            decision_tree_classifier(x_train, y_train, x_val, y_val, x_test, y_test)
        elif choice == '2':
            basic_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
        elif choice == '3':
            developed_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
        elif choice == '4':
            final_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
        elif choice == '5':
            dcgan(x_train, y_train, x_test, y_test)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
