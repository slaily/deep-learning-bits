from keras.datasets import reuters


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10_000)
print('Train data length: ', len(train_data))
print('Test data length: ', len(test_data))
print('Data: ', train_data[10])