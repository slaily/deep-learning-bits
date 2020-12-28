from keras import Input
from keras import layers
from keras.models import Model


text_vocabulary_size = 10_000
question_vocabulary_size = 10_000
answer_vocabulary_size = 500
# The text input is a variable- length sequence of integers.
# Note that you can optionally name the inputs.
text_input = Input(shape=(None,), dtype='int32', name='text')
# Embeds the inputs into a sequence of vectors of size 64
embedded_text = layers.Embedding(
    64, text_vocabulary_size
)(text_input)
# Encodes the vectors in a single vector via an LSTM
encoded_text = layers.LSTM(32)(embedded_text)
# Same process (with different layer instances) for the question
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(
    32, question_vocabulary_size
)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
# Concatenates the encoded question and encoded text
concatenated = layers.concatenate(
    [encoded_text, encoded_question], axis=-1
)
# Adds a softmax classifier on top
answer = layers.Dense(
    answer_vocabulary_size, activation='softmax'
)(concatenate)
# At model instantiation, you specify the two inputs and the output.
model = Model([text_input, question_input], answer)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)
