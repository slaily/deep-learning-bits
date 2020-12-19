from keras.datasets import reuters

(train_data, _), (_, _) = reuters.load_data(num_words=10_000)
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Note that the indices are offset by 3 because
# 0, 1, and 2 are reserved indices for “padding,
# ” “start of sequence,” and “unknown.”
decoded_newswire = ' '.join([reverse_word_index.get(idx - 3, '?') for idx in train_data[0]])
print('Decoded newswire: ', decoded_newswire)