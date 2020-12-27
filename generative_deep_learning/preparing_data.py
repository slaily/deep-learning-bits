"""
Downloading and parsing the initial text file
"""
import keras
import numpy as np


# Downloading and parsing the initial text file
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text = open(path).read.lower()
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
