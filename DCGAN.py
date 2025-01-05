import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

# -------- DCGAN Components -------- #
def build_generator(noise_dim):
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(21 * 21 * 256),  # Adjusted to 21x21 base size
        LeakyReLU(0.2),
        Reshape((21, 21, 256)),  # Start with 21x21x256
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Conv2D(1, kernel_size=3, activation="tanh", padding="same")  # Final output shape (84, 84, 1)
    ])
    return model


def build_discriminator(img_shape):
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(0.2),
        Flatten(),
        Dense(128),
        LeakyReLU(0.2),
        Dense(1, activation="sigmoid")
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# -------- Final CNN Components -------- #
def load_full_images(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    img = tf.keras.preprocessing.image.load_img(file_path, color_mode="grayscale", target_size=(84, 84))
                    images.append(np.array(img))
                    labels.append(int(label_folder))  # Use folder name as label
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_for_cnn(images):
    # Ensure correct shape and normalize
    if images.ndim == 4:  # Already in (batch_size, 84, 84, 1)
        return images / 255.0
    num_images = images.shape[0]
    return images.reshape(num_images, 84, 84, 1) / 255.0

def split_labels(labels):
    return np.array([[int(str(label).zfill(3)[0]), int(str(label).zfill(3)[1]), int(str(label).zfill(3)[2])] for label in labels])

def create_final_cnn():
    base_input = Input(shape=(84, 84, 1))
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(base_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    digit1_output = Dense(10, activation='softmax', name='digit1')(x)
    digit2_output = Dense(10, activation='softmax', name='digit2')(x)
    digit3_output = Dense(10, activation='softmax', name='digit3')(x)
    model = tf.keras.Model(inputs=base_input, outputs=[digit1_output, digit2_output, digit3_output])
    return model

def train_and_evaluate_final_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(
        optimizer='adam',
        loss={
            'digit1': 'sparse_categorical_crossentropy',
            'digit2': 'sparse_categorical_crossentropy',
            'digit3': 'sparse_categorical_crossentropy'
        },
        metrics={
            'digit1': ['accuracy'],
            'digit2': ['accuracy'],
            'digit3': ['accuracy']
        }
    )
    model.fit(
        x_train, {'digit1': y_train[:, 0], 'digit2': y_train[:, 1], 'digit3': y_train[:, 2]},
        validation_data=(x_val, {'digit1': y_val[:, 0], 'digit2': y_val[:, 1], 'digit3': y_val[:, 2]}),
        epochs=10,
        batch_size=32
    )
    y_pred = model.predict(x_test)
    y_pred_classes = [np.argmax(pred, axis=1) for pred in y_pred]
    for i in range(3):
        print(f"F1 score for digit {i+1}: {f1_score(y_test[:, i], y_pred_classes[i], average='macro'):.4f}")

# -------- Main Execution -------- #
# Paths
train_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\train'
val_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\val'
test_folder = r'C:\Users\josep\Documents\GitHub\MachineLearning\dataset2\triple_mnist\test'

# Load and preprocess datasets
x_train, y_train = load_full_images(train_folder)
x_val, y_val = load_full_images(val_folder)
x_test, y_test = load_full_images(test_folder)
x_train_cnn = preprocess_for_cnn(x_train)
x_val_cnn = preprocess_for_cnn(x_val)
x_test_cnn = preprocess_for_cnn(x_test)
y_train_split = split_labels(y_train)
y_val_split = split_labels(y_val)
y_test_split = split_labels(y_test)

# Build and train DCGAN
noise_dim = 100
generator = build_generator(noise_dim)
discriminator = build_discriminator((84, 84, 1))
gan = build_gan(generator, discriminator)
synthetic_images = generator.predict(np.random.normal(0, 1, (10000, noise_dim)))
print(f"Shape of generated synthetic images: {synthetic_images.shape}")
synthetic_images = (synthetic_images + 1) / 2.0  # Rescale to [0, 1]


# Combine synthetic and original data
x_train_augmented = np.concatenate([x_train_cnn, synthetic_images], axis=0)
y_train_augmented = np.concatenate([y_train_split, y_train_split[:len(synthetic_images)]], axis=0)

# Train Final CNN
final_cnn_model = create_final_cnn()
train_and_evaluate_final_cnn(final_cnn_model, x_train_augmented, y_train_augmented, x_val_cnn, y_val_split, x_test_cnn, y_test_split)
