model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e - 5),
    metrics=['acc']
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)