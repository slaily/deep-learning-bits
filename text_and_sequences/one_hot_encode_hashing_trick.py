import numpy as np


samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# Stores the words as vectors of size 1,000. If you have close to 1,000 words
# (or more), you’ll see many hash collisions, which will decrease the accuracy of this encoding method.
dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # Hashes the word into a random integer index between 0 and 1,000
        index = abs(hash(word) % dimensionality)
        results[i, j, index] = 1.

print('Transformed Results: ', results)