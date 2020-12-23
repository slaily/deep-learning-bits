import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Embedding
)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


imdb_dir = '/Users/iliyanslavov/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            file = open(os.path.join(dir_name, fname))
            texts.append(file.read())
            file.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Цuts off reviews after 100 words
maxlen = 100
# Trains on 200 samples
training_samples = 200
# Validates on 10,000 samples
validation_samples = 10_000
# Considers only the top 10,000 words in the dataset
max_words = 10_000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens')
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print(f'Shape of data tensor: {data.shape}')
print(f'Shape of label tensor: {labels.shape}')
# Splits the data into a training set and a validation set,
# but first shuffles the data, because you’re starting with
# data in which samples are ordered (all negative first, then all positive)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
glove_dir = '/Users/iliyanslavov/Downloads/glove.6B'
embeddings_index = {}
file = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

file.close()
print(f'Found {len(embeddings_index)} word vectors.')
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

### Model with pretrained weights ###
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
# # LOADING THE GLOVE EMBEDDINGS IN THE MODEL
# # Loading pretrained word embeddings into the Embedding layer
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
# model.compile(
#     optimizer='rmsprop',
#     loss='binary_crossentropy',
#     metrics=['acc']
# )
# history = model.fit(
#     x_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     validation_data=(x_val, y_val)
# )
# model.save_weights('pre_trained_glove_model.h5')
### Model without pretrained weghts ###
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)
### Plotting ###
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
