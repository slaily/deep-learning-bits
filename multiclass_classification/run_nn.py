import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.datasets import reuters


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


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10_000)
print('Train data length: ', len(train_data))
print('Test data length: ', len(test_data))
print('Data: ', train_data[10])
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Note that the indices are offset by 3 because
# 0, 1, and 2 are reserved indices for “padding,
# ” “start of sequence,” and “unknown.”
decoded_newswire = ' '.join([reverse_word_index.get(idx - 3, '?') for idx in train_data[0]])
print('Decoded newswire: ', decoded_newswire)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Let’s set apart 1,000 samples in the training data to use as a validation set.
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
# Now, let’s train the network for 20 epochs.
history = model.fit(
    partial_x_train,
    partial_y_train,
    # epochs=20,
    epochs=9,
    # batch_size=512,
    batch_size=128,
    validation_data=(x_val, y_val)
)
results = model.evaluate(x_test, one_hot_test_labels)
print('Results: ', results)
# Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Plotting the training and validation accuracy
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Generating predictions on new data
predictions = model.predict(x_test)
print('Sample: ', predictions[0].shape)
print('Coeff in this vector: ', np.sum(predictions[0]))
print('The class with the highest probability: ', np.argmax(predictions[0]))