"""
Preparing data: Vectorize data

Pad your lists so that they all have the same length, turn them into an integer tensor of shape (samples, word_indices), and then use as the first layer in your network a layer capable of handling such integer tensors (the Embedding layer, which we’ll cover in detail later in the book).

OR

One-hot encode your lists to turn them into vectors of 0s and 1s. This would mean, for instance, turning the sequence [3, 5] into a 10,000-dimensional vec- tor that would be all 0s except for indices 3 and 5, which would be 1s. Then you could use as the first layer in your network a Dense layer, capable of handling floating-point vector data.
"""
import numpy as np

from keras.datasets import imdb


def vectorize_sequences(sequences, dimension=10_000):
    # Creates an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))

    for idx, sequence in enumerate(sequences):
        # Sets specific indices of results[i] to 1s
        results[idx, sequence] = 1.

    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# Vectorize the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# Vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')