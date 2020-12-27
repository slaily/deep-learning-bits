callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # Divides the learning rate by 10 when triggered
        patience=10,  # The callback is triggered after the validation loss has stopped improving for 10 epochs.
    )
]
model.fit(
    x,
    y,
    batch_size=32,
    callbacks=callbacks_list,
    validation_data=(x_val, y_val)
)
