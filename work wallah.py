import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
import os

# Function to load images from a directory
def load_images_from_folder(folder, split=False):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path).convert('L')
                if split:
                    img = img.resize((28, 84))
                    img_array = np.array(img)
                    thirds = img_array.shape[1] // 3
                    split_images = [img_array[:, i*thirds:(i+1)*thirds] for i in range(3)]
                    images.append(split_images)
                else:
                    img = img.resize((28, 28))
                    img_array = np.array(img)
                    images.append(img_array)
                labels.append(int(label) % 10)  # Ensure labels are between 0 and 9
    return np.array(images), np.array(labels)

# Function to load and split images
def load_split_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path).convert('L')  # Convert image to grayscale
                img = img.resize((28, 84))  # Resize image to 28x84 for three digits
                img_array = np.array(img)
                thirds = img_array.shape[1] // 3
                split_images = [img_array[:, i*thirds:(i+1)*thirds] for i in range(3)]
                images.append(split_images)
                labels.append(int(label) % 10)  # Ensure labels are between 0 and 9
    return np.array(images), np.array(labels)

# Load your dataset
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Normalize and reshape data
def preprocess_data(x, split=False):
    """
    Normalizes and reshapes data for CNN input.
    If split is True, returns data reshaped for 3-channel inputs.
    """
    x = x / 255.0
    if split:
        return x.reshape((x.shape[0], 3, 28, 28, 1))
    else:
        return x.reshape((x.shape[0], 28, 28, 1))


# Create CNN model
def create_cnn_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create CNN model with regularization and dropout
def create_regularized_cnn_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create CNN model for separate digits
def create_digit_cnn_model():
    digit_model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])

    digit_inputs = [Input(shape=(28, 28, 1)) for _ in range(3)]
    digit_outputs = [digit_model(digit_input) for digit_input in digit_inputs]

    concatenated = layers.concatenate(digit_outputs)
    output = layers.Dense(10, activation='softmax')(concatenated)

    combined_model = models.Model(inputs=digit_inputs, outputs=output)
    return combined_model

# Train and evaluate CNN model
def train_and_evaluate_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc:.2f}')
    return history

# Plot learning loss and accuracy
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
def split_and_reshape(imgs):
    """
    Splits each image into three parts (columns) and reshapes each part to (28, 28, 1).
    Ensures the output is a 5D array with dimensions (n_samples, 3, 28, 28, 1).
    """
    parts = []
    for img in imgs:
        split_imgs = np.split(img, 3, axis=1)  # Split each image into 3 columns
        for part in split_imgs:
            if part.shape == (28, 28):  # Ensure the part has correct dimensions
                parts.append(part.reshape((28, 28, 1)))
    return np.array(parts).reshape((-1, 3, 28, 28, 1))


def build_generator():
    model = models.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def train_dcgan():
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    generator = build_generator()

    discriminator.trainable = False

    gan_input = Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)

    gan.compile(optimizer='adam', loss='binary_crossentropy')

    batch_size = 128
    epochs = 10000
    sample_interval = 200

    train_images = preprocess_images(train_folder)  # Ensure images are preprocessed correctly

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_imgs = train_images[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [D accuracy: {100 * d_loss[1]}%] [G loss: {g_loss}]")

            # Save generated images
            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, 100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(f"generated_{epoch}.png")
            plt.close()


        # Logistic Regression
def logistic_regression(x_train_flattened, y_train, x_test_flattened, y_test):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train_flattened, y_train)

    y_test_pred = log_reg_model.predict(x_test_flattened)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy: {test_acc:.2f}')
    print(classification_report(y_test, y_test_pred))


def main():
    print("Select the model to run:")
    print("1. CNN Iteration 1: Whole Image")
    print("2. CNN Iteration 2: Separate Digits")
    print("3. CNN Iteration 3: Data Regularization and Augmentation")
    print("4. DCGAN")
    print("5. Logistic Regression")

    choice = int(input("Enter the number of your choice: "))

    if choice == 1 or choice == 5:
        x_train, y_train = load_images_from_folder(train_folder)
        x_val, y_val = load_images_from_folder(val_folder)
        x_test, y_test = load_images_from_folder(test_folder)
    elif choice == 2 or choice == 3:
        x_train, y_train = load_split_images_from_folder(train_folder)
        x_val, y_val = load_split_images_from_folder(val_folder)
        x_test, y_test = load_split_images_from_folder(test_folder)

    if choice == 1:
        print("Running CNN Iteration 1: Whole Image")
        x_train_p, x_val_p, x_test_p = preprocess_data(x_train), preprocess_data(x_val), preprocess_data(x_test)
        model = create_cnn_model()
        history = train_and_evaluate_cnn(model, x_train_p, y_train, x_val_p, y_val, x_test_p, y_test)
        plot_history(history)
    elif choice == 2:
        print("Running CNN Iteration 2: Separate Digits")
        
        # Normalize the data and split into three parts
        x_train_p = split_and_reshape(x_train)  # Pass original x_train for splitting
        x_val_p = split_and_reshape(x_val)
        x_test_p = split_and_reshape(x_test)

        # Create the CNN model for each digit
        model = create_digit_cnn_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(
            [x_train_p[:, i, :, :, :] for i in range(3)], y_train,  # Pass slices for 3 inputs
            epochs=10,
            validation_data=([x_val_p[:, i, :, :, :] for i in range(3)], y_val)
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(
            [x_test_p[:, i, :, :, :] for i in range(3)], y_test, verbose=2
        )
        print(f'Test accuracy: {test_acc:.2f}')
        plot_history(history)

    elif choice == 3:
        print("Running CNN Iteration 3: Data Regularization and Augmentation with Split Images")

        # Normalize the data and split into three parts
        x_train_p = split_and_reshape(x_train)  # Pass original x_train for splitting
        x_val_p = split_and_reshape(x_val)
        x_test_p = split_and_reshape(x_test)

        # Flatten the data for augmentation
        x_train_flat = x_train_p.reshape(-1, 28, 28, 1)
        y_train_tiled = np.repeat(y_train, 3)  # Match the repeated image data length

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3]
        )
        datagen.fit(x_train_flat)  # Fit on the flattened array

        # Create a regularized CNN model
        model = create_regularized_cnn_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Training the model with augmented data
        history = model.fit(
            datagen.flow(x_train_flat, y_train_tiled, batch_size=32),
            epochs=50,
            validation_data=(x_val_p.reshape(-1, 28, 28, 1), np.repeat(y_val, 3)),
            callbacks=[early_stopping]
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(
            x_test_p.reshape(-1, 28, 28, 1), np.repeat(y_test, 3), verbose=2
        )
        print(f'Test accuracy: {test_acc:.2f}')
        plot_history(history)

    elif choice == 4:
        print("Running DCGAN")
        train_dcgan()
    elif choice == 5:
        print("Running Logistic Regression")
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(x_train_flattened, y_train)
        y_test_pred = log_reg_model.predict(x_test_flattened)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f'Test accuracy: {test_acc:.2f}')
        print(classification_report(y_test, y_test_pred))
    else:
        print("Invalid choice. Please select a valid option from the menu.")
        exit()

if __name__ == "__main__":
    while True:
        main()

