import os

from json import dump

import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


### Data Augmentation ###
# train_cats_dir = '/Users/iliyanslavov/Downloads/cats_and_dogs_small/train/cats'
# # Setting up a data augmentation configuration via ImageDataGenerator
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# # Displaying some randomly augmented training images
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# img_path = fnames[3]
# # Reads the image and resizes it
# img = image.load_img(img_path, target_size=(150, 150))
# # Converts it to a Numpy array with shape (150, 150, 3)
# x = image.img_to_array(img)
# # Reshape it to (1, 150, 150, 3)
# x = x.reshape((1,) + x.shape)
# i = 0
# # Generates batches of randomly transformed images.
# # Loops indefinitely, so you need to break the loop at some point!
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1

#     if i % 4 == 0:
#         break

# plt.show()
### Model ####
model = models.Sequential()
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(150, 150, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)
print("Model summary: ", model.summary())
### Train using Data augmentation and dropout ###
train_dir = '/Users/iliyanslavov/Downloads/cats_and_dogs_small/train'
validation_dir = '/Users/iliyanslavov/Downloads/cats_and_dogs_small/validation'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Note that the validation data shouldn’t be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
history = model.fit_generator(
    train_generator,
    # steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)
model.save('cats_and_dogs_small_3_using_data_augmentation.h5')
with open('history_nn_v3.json', 'w', encoding='utf-8') as f:
    dump(history.history, f, ensure_ascii=False, indent=4)
### Plotting ###
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
