"""
You’ll get an input image—a picture of a cat, not part of the images the network was trained on.
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image


img_path = '/Users/iliyanslavov/Downloads/dataset-dogs-vs-cats/test/cats/cat.1700.jpg'
# Preprocesses the image into a 4D tensor
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs that were preprocessed this way.
img_tensor /= 255.
print(img_tensor.shape)
# Displaying the test picture
plt.imshow(img_tensor[0])
plt.show()
