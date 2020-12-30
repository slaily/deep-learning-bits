"""
GAN generator network
"""
import keras
import numpy as np

from keras import layers


latend_dim = 32
height = 32
width = 32
channels = 3
generator_input = keras.Input(shape=(latend_dim,))
# Transforms the input into a 16 × 16 128-channel feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# Upsamples to 32 × 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# Produces a 32 × 32 1-channel feature map (shape of a CIFAR10 image)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
# Instantiates the generator model, which maps the input
# of shape (latent_dim,) into an image of shape (32, 32, 3)
generator = keras.models.Model(generator_input, x)
generator.summary()
