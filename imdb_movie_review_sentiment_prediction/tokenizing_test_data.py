"""
Tokenizing the data of the test set
"""
import os


imdb_dir = '/Users/iliyanslavov/Downloads/aclImdb'
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)

    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            file = open(os.path.join(dir_name, fname))
            texts.append(file.read())
            file.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
