import random


k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []

for fold in range(k):
    # Selects the validation- data partition
    validation_data = data[
        num_validation_samples * fold: num_validation_samples * (fold + 1)
    ]
    # Uses the remainder of the data as training data.
    # Note that the + operator is list concatenation, not summation.
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    # Creates a brand-new instance of the model (untrained)
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

# Validation score: average of the validation scores of the k folds
validation_score = np.average(validation_scores)
# Trains the final model on all non- test data available
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)