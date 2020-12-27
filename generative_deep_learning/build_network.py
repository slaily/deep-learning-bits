from keras import layers


# Single-layer LSTM model for next-character prediction
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
# Model compilation configuration
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# Function to sample the next character given the model’s predictions


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinominal(1, preds, 1)

    return np.argmax(probas)


# Text-generation loop
import sys
import random


# Trains the model for 60 epochs
for epoch in range(1, 60):
    print(f'Epoch: {epoch}')
    model.fit(x, y, batch_size=128, epochs=1)
    # Selects a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print(f'--- Generating with seed: {generated_text} ---')

    # Tries a range of different sampling temperatures
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f'--- Temperature {temperature} ---')
        sys.stdout.write(generated_text)

        # Generates 400 characters, starting from the seed text
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # Samples the next character
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
