"""
Data preprocessing
As you know by now, data should be formatted into appropriately preprocessed floating- point tensors before being fed into the network. Currently, the data sits on a drive as JPEG files, so the steps for getting it into the network are roughly as follows:
    1 Read the picture files.
    2 Decode the JPEG content to RGB grids of pixels.
    3 Convert these into floating-point tensors.
    4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know,
    neural networks prefer to deal with small input values).
"""
from keras.preprocessing.image import ImageDataGenerator


# Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/Users/iliyanslavov/Downloads/cats_and_dogs_small/train',
    target_size=(150, 150),  # Resizes all images to 150 × 150
    batch_size=20,
    class_mode='binary'  # Because you use binary_crossentropy loss, you need binary labels.
)
validation_generator = test_datagen.flow_from_directory(
    '/Users/iliyanslavov/Downloads/cats_and_dogs_small/validation',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('Data Batch Shape:', data_batch.shape)
    print('Labels Batch Shape:', labels_batch.shape)
    break
