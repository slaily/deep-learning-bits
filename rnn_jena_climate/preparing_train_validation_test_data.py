"""
Now, let’s use the abstract generator function to instantiate three generators: one for training, one for validation, and one for testing. Each will look at different temporal segments of the original data: the training generator looks at the first 200,000 time- steps, the validation generator looks at the following 100,000, and the test generator looks at the remainder.
"""
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size
)
val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size
)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
    batch_size=batch_size
)
# How many steps to draw from val_gen in
# order to see the entire validation set
val_steps = (300000 - 200001 - lookback)
# How many steps to draw from test_gen in
# order to see the entire test set
test_steps = (len(float_data) - 300001 - lookback)
