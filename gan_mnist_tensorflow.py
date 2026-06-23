"""
Simple GAN for MNIST Image Generation

Run:
    python gan_mnist_tensorflow.py

This script trains a basic GAN on MNIST and saves generated images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 20


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

seed = tf.random.normal([16, LATENT_DIM])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def save_generated_images(epoch):
    predictions = generator(seed, training=False)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap="gray")
        plt.axis("off")

    output_path = os.path.join(RESULTS_DIR, f"gan_generated_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved generated images to {output_path}")


def main():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(60000)
        .batch(BATCH_SIZE, drop_remainder=True)
    )

    for epoch in range(1, EPOCHS + 1):
        gen_losses = []
        disc_losses = []

        for image_batch in train_dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(float(gen_loss))
            disc_losses.append(float(disc_loss))

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Generator Loss: {np.mean(gen_losses):.4f} | "
            f"Discriminator Loss: {np.mean(disc_losses):.4f}"
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            save_generated_images(epoch)

    save_generated_images(EPOCHS)


if __name__ == "__main__":
    main()
