import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


# Load and preprocess data
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_train = (X_train / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
X_train = np.clip(X_train, -1, 1).astype(np.float32)

print('X_train shape:', X_train.shape)

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 100
smooth = 0.1
learning_rate = 0.0002

# Weight initialization
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,), kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", use_bias=False, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=1, padding="same", activation="tanh", kernel_initializer=initializer)
])
generator.summary()

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=(28, 28, 1), kernel_initializer=initializer),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding="same", kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=initializer)
])
discriminator.summary()

# Compile models
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
discriminator.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"])
discriminator.trainable = False

# Combined GAN model
z = tf.keras.layers.Input(shape=(latent_dim,))
img = generator(z)
decision = discriminator(img)
dcgan = tf.keras.models.Model(inputs=z, outputs=decision)
dcgan.compile(optimizer=optimizer, loss="binary_crossentropy")
dcgan.summary()

# Training loop
d_loss, g_loss = [], []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):
        # Train Discriminator
        discriminator.trainable = True

        # Real samples
        real_imgs = X_train[i * batch_size: (i + 1) * batch_size]
        real_labels = np.ones((batch_size, 1)) * (1 - smooth)
        print("Training discriminator on real data...")
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

        # Fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict_on_batch(noise)
        fake_labels = np.zeros((batch_size, 1))
        print("Training discriminator on fake data...")
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Train Generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        print("Training generator...")
        g_loss_batch = dcgan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Append losses
    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch)

    print(
        f"Epoch {e}/{epochs} | Discriminator Loss: {d_loss[-1]:.4f} | Generator Loss: {g_loss[-1]:.4f}"
    )


    # Store losses
    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch)

    # Print progress
    print(
        f"Epoch {e}/{epochs}, Discriminator Loss: {d_loss[-1]:.4f}, Generator Loss: {g_loss[-1]:.4f}",
        end="\r"
    )

    # Generate and visualize images every 10 epochs
    if e % 10 == 0:
        samples = 10
        noise = np.random.normal(0, 1, (samples, latent_dim))
        generated_imgs = generator.predict(noise)

        fig, axes = plt.subplots(1, samples, figsize=(15, 3))
        for ax, img in zip(axes, generated_imgs):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# Plot loss curves
plt.plot(d_loss, label='Discriminator Loss')
plt.plot(g_loss, label='Generator Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
