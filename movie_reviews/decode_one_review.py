from keras.datasets import imdb


(train_data, _), (_, _) = imdb.load_data(num_words=10000)
words_indexes = imdb.get_word_index()
# Reverses it, mapping integer indices to words
reversed_words_indexes = dict(
    [
        (value, key) for (key, value) in words_indexes.items()
    ]
)
# Decodes the review.
# Note that the indices are offset by 3 because 0, 1, and 2
# are reserved indices for “padding,” “start of sequence,” and “unknown.”
decoded_review = ' '.join(
    [reversed_words_indexes.get(i - 3, '?') for i in train_data[0]]
)
print('--- Decoded Review ---')
print(decoded_review)
print('----------------------')