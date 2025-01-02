import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Generator Model
def build_generator(noise_dim):
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(7 * 21 * 256),
        LeakyReLU(negative_slope=0.2),
        Reshape((7, 21, 256)),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2D(1, kernel_size=3, activation="tanh", padding="same")
    ])
    return model

# Discriminator Model
def build_discriminator(img_shape):
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Flatten(),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(1, activation="sigmoid")
    ])
    return model

# Assemble GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Save Generated Images
def save_generated_images(epoch, generator, noise_dim, examples=5, dim=(1, 5), figsize=(15, 2)):
    noise = np.random.normal(0, 1, (examples, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(gen_imgs[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"triple_mnist_generated_epoch_{epoch}.png")
    plt.close()

# Training Function
def train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim, save_interval):
    valid = np.ones((batch_size, 1)) * 0.9  # Label smoothing
    fake = np.zeros((batch_size, 1)) * 0.1

    for epoch in range(epochs):
        # Train Discriminator
        discriminator.trainable = True
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_imgs = train_data[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Safely handle printing losses
        d_loss_val = d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss
        g_loss_val = g_loss[0] if isinstance(g_loss, (list, np.ndarray)) else g_loss

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss_val:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss_val:.4f}]")

        # Save generated images at intervals
        if (epoch + 1) % save_interval == 0:
            save_generated_images(epoch + 1, generator, noise_dim)

# Load and preprocess Triple MNIST data
def load_triple_mnist():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

    # Create Triple MNIST data by horizontally stacking 3 random images
    triple_data = []
    for _ in range(len(x_train)):
        digits = np.random.choice(len(x_train), 3, replace=False)
        triple_img = np.hstack([x_train[digits[0]], x_train[digits[1]], x_train[digits[2]]])
        triple_data.append(triple_img)

    triple_data = np.expand_dims(np.array(triple_data), axis=-1)
    return triple_data

# Hyperparameters
img_shape = (28, 84, 1)  # Adjusted for 28x84 combined images
noise_dim = 100
epochs = 5000
batch_size = 32
save_interval = 500

# Build and compile models
generator = build_generator(noise_dim)
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy", metrics=["accuracy"])

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

# Load data and train the DCGAN
train_data = load_triple_mnist()
train_dcgan(generator, discriminator, gan, train_data, epochs, batch_size, noise_dim, save_interval)
