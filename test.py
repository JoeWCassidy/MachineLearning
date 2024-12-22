import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
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
    history = model.fit(x_train_flat, y_train, validation_data=(x_val_flat, y_val), epochs=3)
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
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
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
    history = model.fit(datagen.flow(x_train, y_train), validation_data=(x_val, y_val), epochs=10)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Final CNN Test Accuracy: {test_acc}, Test F1 Score: {test_f1}")
    plot_training_curves(history)

def dcgan(x_train):
    print("Training DCGAN...")
    noise_dim = 100
    img_shape = x_train.shape[1:]

    generator = Sequential([
        Dense(256, activation='relu', input_dim=noise_dim),
        Dense(512, activation='relu'),
        Dense(np.prod(img_shape), activation='sigmoid'),
        layers.Reshape(img_shape)
    ])

    discriminator = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False

    gan = Sequential([generator, discriminator])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    epochs = 10000
    batch_size = 256
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_images = generator.predict(noise)
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    # Generate synthetic images and visualize
    noise = np.random.normal(0, 1, (10, noise_dim))
    synthetic_images = generator.predict(noise)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(synthetic_images[i].reshape(x_train.shape[1:2]), cmap='gray')
        plt.axis('off')
    plt.show()
##
##def main_menu():
##    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_triple_mnist()
##    while True:
##        print("\nMain Menu")
##        print("1. Decision Tree")
##        print("2. Basic CNN")
##        print("3. Slightly Developed CNN")
##        print("4. Final CNN")
##        print("5. DCGAN")
##        print("6. Exit")
##        choice = input("Enter your choice: ")
##
##        if choice == '1':
##            decision_tree_classifier(x_train, y_train, x_val, y_val, x_test, y_test)
##        elif choice == '2':
##            basic_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
##        elif choice == '3':
##            developed_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
##        elif choice == '4':
##            final_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
##        elif choice == '5':
##            dcgan(x_train)
##        elif choice == '6':
##            print("Exiting...")
##            break
##        else:
##            print("Invalid choice. Please try again.")
##
##if __name__ == "__main__":
##    main_menu()

decision_tree_classifier(x_train, y_train, x_val, y_val, x_test, y_test)
basic_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
developed_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
final_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
dcgan(x_train)
