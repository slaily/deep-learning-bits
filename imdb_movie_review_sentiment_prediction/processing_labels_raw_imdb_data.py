"""
Processing the labels of the raw IMDB data
"""
import os


imdb_dir = '/Users/iliyanslavov/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            file = open(os.path.join(dir_name, fname))
            texts.append(file.read())
            file.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
