import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_images(target_label, batch_size):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    images = np.concatenate((train_images, test_images), axis=0)
    
    if target_label is not None:
        labels = np.concatenate((train_labels, test_labels), axis=0)
        images = images[labels == target_label] 
    
    images = images.reshape(len(images), 28, 28, 1).astype('float32') / 255.0
    rounded_size = (len(images) // batch_size) * batch_size
    images = images[:rounded_size, :, :, :]
    
    prep_config = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5.0,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1)
    
    return prep_config.flow(images, batch_size=batch_size, shuffle=True)


def display_images(images, save_path):
    images = images.numpy()
    
    plt.figure(figsize=(len(images), 1))
    for img_id, image in enumerate(images):
        plt.subplot(1, len(images), img_id + 1)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()


def create_generator(noise_dim, num_filters):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(6 * 6 * num_filters, input_shape=(noise_dim,)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((6, 6, num_filters)),
        tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=4, strides=2),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=4, strides=2),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, activation='sigmoid')
    ])


def create_discriminator(num_filters):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, input_shape=(28, 28, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(num_filters, kernel_size=4, strides=2),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(num_filters, kernel_size=4, strides=2),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation=None)
    ])


@tf.function    
def generate_fake_images(num_to_generate, noise_dim, generator):
    noise = tf.random.normal(shape=(num_to_generate, noise_dim))
    fake_images = generator(noise)
    return fake_images


def wasserstein_loss(scores, target_label):
    if target_label:
        return -tf.reduce_mean(scores)
    else:
        return tf.reduce_mean(scores)


@tf.function
def train_generator(batch_size, noise_dim, generator, discriminator, gen_optimizer):
    with tf.GradientTape() as tape:
        fake_images = generate_fake_images(batch_size, noise_dim, generator)
        fake_scores = discriminator(fake_images)
        loss = wasserstein_loss(fake_scores, True)
    
    gen_gradients = tape.gradient(loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))


@tf.function
def train_discriminator(real_images, batch_size, noise_dim, generator, discriminator,
                        disc_optimizer, grad_pen_weight):
    with tf.GradientTape() as tape:
        fake_images = generate_fake_images(batch_size, noise_dim, generator)
        real_scores = discriminator(real_images)
        fake_scores = discriminator(fake_images)
        loss = wasserstein_loss(real_scores, True) + wasserstein_loss(fake_scores, False)

        eps = tf.random.uniform(shape=(batch_size, 1, 1, 1))
        mid_images = eps * real_images + (1.0 - eps) * fake_images
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(mid_images)
            mid_scores = discriminator(mid_images)
        mid_score_grad = inner_tape.gradient(mid_scores, mid_images)
        mid_score_grad_norm = tf.math.sqrt(
                tf.reduce_sum(tf.square(mid_score_grad), axis=(1, 2, 3)))
        gradient_penalty = tf.reduce_mean(tf.square(mid_score_grad_norm - 1.0))
        loss += grad_pen_weight * gradient_penalty
    
    disc_gradients = tape.gradient(loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))


def run_gan(target_label, num_train_steps, batch_size, noise_dim, num_filters,
            gen_learning_rate, disc_learning_rate, disc_steps_per_gen_step,
            grad_pen_weight, demo_size, save_path):
    '''
    Train standard or Wasserstein-GP GAN to generate handwritten digits. 
    
    :param target_label: the digit to generate (if None, will generate all digits)
    :param num_train_steps: total number of training steps to perform
    :param batch_size: number of training samples per batch
    :param noise_dim: dimensionality of the latent noise space
    :param num_filters: number of convolutional filters to use in up/down layers
    :param gen_learning_rate: learning rate for training generator
    :param disc_learning_rate: learning rate for training discriminator
    :param disc_steps_per_gen_step: number of discriminator training steps to perform
                               in between each generator training step
    :param grad_pen_weight: relative weight for gradient penalty in loss function
    :param demo_size: number of sample generated images to plot
    :param save_path: location to save generated images to (if None, will not save images)
    '''    
    dataset = load_images(target_label, batch_size)
    
    generator = create_generator(noise_dim, num_filters)
    discriminator = create_discriminator(num_filters)
    
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_learning_rate)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_learning_rate)
    
    gen_step_freq = disc_steps_per_gen_step + 1

    for batch_id, real_images in zip(range(1, num_train_steps * gen_step_freq + 1), dataset):
        assert tf.shape(real_images).numpy()[0] == batch_size
        
        if batch_id % gen_step_freq != 0:
            train_discriminator(real_images, batch_size, noise_dim, generator,
                                discriminator, disc_optimizer, grad_pen_weight)
        else:
            train_generator(batch_size, noise_dim, generator, discriminator, gen_optimizer)
    
    fake_images = generate_fake_images(demo_size, noise_dim, generator)
    display_images(fake_images, save_path)


if __name__ == '__main__':
    target_label = int(sys.argv[1])
    save_path = sys.argv[2]
    
    run_gan(target_label=target_label,
            num_train_steps=10000,
            batch_size=64,
            noise_dim=16,
            num_filters=32,
            gen_learning_rate = 1.0e-3,
            disc_learning_rate = 1.0e-3,
            disc_steps_per_gen_step = 5,
            grad_pen_weight = 10.0,
            demo_size=8,
            save_path=save_path)
