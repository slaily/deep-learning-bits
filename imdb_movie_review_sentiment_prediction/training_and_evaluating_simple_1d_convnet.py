"""
Training and evaluating a simple 1D convnet on the IMDB data
"""
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop


model = Sequential()
model.add(
    layers.Embedding(max_features, 128, input_length=max_len)
)
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(
    optimizer=RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
