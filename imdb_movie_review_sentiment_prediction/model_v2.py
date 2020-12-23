from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Embedding
)


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()