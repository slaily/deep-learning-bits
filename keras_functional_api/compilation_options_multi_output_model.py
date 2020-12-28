"""
Compilation options of a multi-output model: multiple losses
"""
model.compile(
    optimizer='rmsprop',
    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy']
)
# Equivalent (possible only if you give names to the output layers)
model.compile(
    optimizer='rmsprop',
    loss={
        'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'
    }
)
