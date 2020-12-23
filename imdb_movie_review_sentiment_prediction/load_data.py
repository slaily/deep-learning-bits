from keras.datasets import imdb
from keras import preprocessing


# Number of words to consider as features
max_features = 10000
# Cuts off the text after this number of words
# (among the max_features most common words)
maxlen = 20
# Loads the data as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)
# Turns the lists of integers into a 2D integer
# tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequence(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequence(x_test, maxlen=maxlen)
