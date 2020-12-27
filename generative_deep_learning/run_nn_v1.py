import sys
import random

import keras
import numpy as np

from keras import layers


# original_distribution is a 1D Numpy array of probability values
# that must sum to 1. temperature is a factor quantifying the entropy
# of the output distribution.
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)

    # Returns a reweighted version of the original distribution.
    # The sum of the distribution may no longer be 1,so you divide
    # it by its sum to obtain the new distribution.
    return distribution / np.sum(distribution)


# Downloading and parsing the initial text file
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text = open(path).read().lower()
print('Corpus length: ', len(text))
# Vectorizing sequences of characters
# You’ll extract sequences of 60 characters.
maxlen = 60
# You’ll sample a new sequence every three characters.
step = 3
# Holds the extracted sequences
sentences = []
# Holds the targets (the follow-up characters)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print(f'Number of sequences: {len(sentences)}')
# List of unique characters in the corpus
chars = sorted(list(set(text)))
print(f'Unique characters: {len(chars)}')
# Dictionary that maps unique characters to their index in the list “chars”
char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')
# One-hot encodes the characters into binary arrays
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1



# Single-layer LSTM model for next-character prediction
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
# Model compilation configuration
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# Function to sample the next character given the model’s predictions


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


# Text-generation loop
# Trains the model for 60 epochs
for epoch in range(1, 60):
    print(f'Epoch: {epoch}')
    model.fit(x, y, batch_size=128, epochs=1)
    # Selects a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print(f'--- Generating with seed: {generated_text} ---')

    # Tries a range of different sampling temperatures
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f'--- Temperature {temperature} ---')
        sys.stdout.write(generated_text)

        # Generates 400 characters, starting from the seed text
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # Samples the next character
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
