from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop


model = Sequential()
model.add(
    layers.GRU(32, input_shape=(None, float_data.shape[-1]))
)
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps
)
