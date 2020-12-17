import random


num_validation_samples = 10000
# Shuffling the data is usually appropriate.
np.random.shuffle(data)
# Defines the validation set
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]
# Defines the training set
training_data = data[:]
# Trains a model on the training
# data, and evaluates it on the validation data
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)
# At this point you can tune your model,
# retrain it, evaluate it, tune it again...
# Once you’ve tuned your hyperparameters, it’s common to train
# your final model from scratch on all non-test data available.
model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)

