"""
PREPROCESSING THE EMBEDDINGS

Parsing the GloVe word-embeddings file
"""
import os


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
