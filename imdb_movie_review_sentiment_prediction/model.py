from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten
)


model = Sequential()
# Specifies the maximum input length to the Embedding layer
# so you can later flatten the embedded inputs. After the Embedding
# layer, the activations have shape (samples, maxlen, 8).
model.add(Embedding(10000, 8, input_length=maxlen))
# Flattens the 3D tensor of embeddings into a 2D tensor of
# shape (samples, maxlen * 8)
model.add(Flatten())
# Adds the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
