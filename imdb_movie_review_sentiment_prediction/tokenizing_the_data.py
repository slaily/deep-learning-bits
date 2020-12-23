"""
Tokenizing the text of the raw IMDB data
"""
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# uts off reviews after 100 words
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
