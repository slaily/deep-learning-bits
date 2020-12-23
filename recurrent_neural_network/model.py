from keras.models import Sequential
from keras.layers import (
    Dense,
    Embedding,
    SimpleRNN
)


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
