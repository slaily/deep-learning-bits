"""
Copying images to training, validation, and test directories
"""
import os
import shutil


# Path to the directory where the Directory
# the original dataset was uncompressed
original_dataset_dir = '/Users/iliyanslavov/Downloads/dataset-dogs-vs-cats'
original_dataset_train_dir = os.path.join(original_dataset_dir + '/train', 'train')
original_dataset_test_dir = os.path.join(original_dataset_dir + '/test', 'test')
# Directory where you’ll store your smaller dataset
base_dir = '/Users/iliyanslavov/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)
# Directory for the training split
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
# Directory for the validation split
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# Directory for the test split
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# Directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# Directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# Copies the first 1,000 cat images to train_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# Copies the next 500 cat images to validation_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1501, 2000)]
# Copies the next 500 cat images to test_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_test_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# Copies the first 1,000 dog images to train_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# Copies the next 500 dog images to validation_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1501, 2000)]
# Copies the next 500 dog images to test_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_test_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('Total training cat images: ', len(os.listdir(train_cats_dir)))
print('Total training dog images: ', len(os.listdir(train_dogs_dir)))
print('Total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('Total validation dog images: ', len(os.listdir(validation_dogs_dir)))
print('Total test cat images: ', len(os.listdir(test_cats_dir)))
print('Total test dog images: ', len(os.listdir(test_dogs_dir)))
