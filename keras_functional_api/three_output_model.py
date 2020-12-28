"""
Implementation of a three-output model
"""
from keras import Input
from keras import layers
from keras.models import Model


vocabulary_size = 50_000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()x
x = layers.Dense(128, activation='relu')(x)
# Note that the output layers are given names.
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(
    num_income_groups,
    activation='softmax',
    name='income'
)(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(
    posts_input,
    [age_prediction, income_prediction, gender_prediction]
)
