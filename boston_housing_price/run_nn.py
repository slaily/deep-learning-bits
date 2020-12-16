import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.datasets import boston_housing


def build_model():
    """
    Because so few samples are available, you’ll use a very small network with two hidden layers, each with 64 units. In general, the less training data you have, the worse overfit- ting will be, and using a small network is one way to mitigate overfitting.
    """
    model = models.Sequential()
    model.add(layers.Dense(
        64,
        activation='relu',
        input_shape=(train_data.shape[1],)
    ))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])

    return model


def smooth_curve(points, factor=0.9):
    smoothed_points = []

    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print('Train data shape: ', train_data.shape)
print('Test data shape: ', test_data.shape)
print('Train targets: ', train_targets)
# Normalizing the data
# A widespread best practice to deal with such data is to do feature-wise normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /=std
# Validating your approach using K-fold validation
k = 4
num_val_samples = len(train_data) // 4
num_epochs = 100
# num_epochs = 500
all_scores = []
all_mae_histories = []

for number in range(k):
    print('Processing fold #', number)
    # Prepares the validation data: data from partition #k
    val_data = train_data[number * num_val_samples: (number + 1) * num_val_samples]
    val_targets = train_targets[number * num_val_samples: (number + 1) * num_val_samples]
    # Prepares the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [
            train_data[:number * num_val_samples],
            train_data[(number + 1) * num_val_samples:]
        ],
        axis=0
    )
    partial_train_targets = np.concatenate(

        [
            train_targets[:number * num_val_samples],
            train_targets[(number + 1) * num_val_samples:]
        ],
        axis=0
    )
    # Builds the Keras model (already compiled)
    model = build_model()
    # --- First fir version ---
    # Trains the model (in silent mode, verbose = 0)
    # model.fit(
    #     partial_train_data,
    #     partial_train_targets,
    #     epochs=num_epochs,
    #     batch_size=1,
    #     verbose=0
    # )
    # --- End First fir version ---
    # Evaluates the model on the validation data
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)
    # --- Another fit version ---
    # Saving the validation logs at each fold
    # history = model.fit(
    #     partial_train_data,
    #     partial_train_targets,
    #     validation_data=(val_data, val_targets),
    #     epochs=num_epochs,
    #     batch_size=1,
    #     verbose=0
    # )
    # mae_history = history.history['val_mean_absolute_error']
    # all_mae_histories.append(mae_history)
    # Building the history of successive mean K-fold validation scores
    # average_mae_history = [
    #     np.mean([x[idx] for x in all_mae_histories]) for idx in range(num_epochs)
    # ]
    # --- Training the final model ---
    # Trains it on the entirety of the data
    model.fit(
        train_data,
        train_targets,
        epochs=80,
        batch_size=16,
        verbose=0
    )

# Building your network
# print('All scores: ', all_scores)
# print('MEAN scores: ', np.mean(all_scores))
# Plotting validation scores
# smooth_mae_history = smooth_curve(average_mae_history[10:])
# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
# --- Training the final model score ---
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('Test MSE Score: ', test_mse_score)
print('Test MAE Score: ', test_mae_score)