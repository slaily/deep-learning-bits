import numpy as np


num_samples = 1_000
max_length = 100
# Generates dummy Numpy data
text = np.random.randint(
    1, text_vocabulary_size, size=(num_samples, max_length)
)
question = np.random.randint(
    1, question_vocabulary_size, size(num_samples, max_length)
)
# Answers are one- hot encoded, not integers
answers = np.random.randint(
    0, 1, size=(num_samples, answer_vocabulary_size)
)
#  Fitting using a list of inputs
model.fit([text, question], answers, epochs=10, batch_size=128)
# Fitting using a dictionary of inputs (only if inputs are named)
model.fit(
    ('text': text, 'question': question),
    answers,
    epochs=10,
    batch_size=128
)
