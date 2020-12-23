"""
Preparing the IMDB data
"""
from keras.datasets import imdb
from keras.preprocessing import sequence


# Number of words to consider as features
max_features = 10_000
# Cuts off texts after this many words
# (among the max_features most common words)
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_features
)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print(f'input_train shape: {input_train.shape}')
print(f'input_test shape: {input_test.shape}')