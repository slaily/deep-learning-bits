import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for idx, sequence in enumerate(sequences):
        results[idx, sequence] = 1.

    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))

    for idx, label in enumerate(labels):
        results[idx, label] = 1.

    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)