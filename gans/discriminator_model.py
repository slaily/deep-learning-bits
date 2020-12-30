"""
The GAN discriminator network
"""
import keras
import numpy as np

from keras import layers


discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
# One dropout layer: an important trick!
x = layers.Dropout(0.4)(x)
# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)
# Instantiates the discriminator model, which turns a (32, 32, 3)
# input into a binary classifi-cation decision (fake/real)
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,  # Uses gradient clipping (by value) in the optimizer
    decay=1e-8  # To stabilize training, uses learning-rate decay
)
discriminator.compile(
    optimizer=discriminator_optimizer,
    loss='binary_crossentropy'
)
