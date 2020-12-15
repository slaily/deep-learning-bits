import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions


# Note that you include the densely connected classifier on top;
# in all previous cases, you discarded it.
model = VGG16(weights='imagenet')
# Local path to the target image
img_path = '/Users/iliyanslavov/Downloads/creative_commons_elephant.jpg'
# Python Imaging Library (PIL) image of size 224 × 224
img = image.load_img(img_path, target_size=(224, 224))
# float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)
# Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# Preprocesses the batch (this does channel-wise color normalization)
x = preprocess_input(x)
# You can now run the pretrained network on the image and decode its
# prediction vec- tor back to a human-readable format.
preds = model.predict(x)
print('Predicted: ', decode_predictions(preds, top=3)[0])
# “African elephant” class, at index...
np.argmax(preds[0])
# To visualize which parts of the image are the most African elephant–like,
# let’s set up the Grad-CAM process.
# Setting up the Grad-CAM algorithm
# “African elephant” entry in the prediction vector
african_elephant_output = model.output[:, 386]
# Output feature map of the block5_conv3 layer, the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')
# Gradient of the “African elephant” class with regard to the output feature map of block5_conv3
# grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
grads = tf.GradientTape().gradient(african_elephant_output, last_conv_layer.output)[0]
# Vector of shape (512,), where each entry is the mean intensity
# of the gradient over a specific feature-map channe
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# Lets you access the values of the quantities you just defined:
# pooled_grads and the output feature map of block5_conv3, given a sample image
iterate = K.function(
    [model.input],
    [pooled_grads, last_conv_layer.output[0]]
)
# Values of these two quantities, as Numpy arrays,
# given the sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# Multiplies each channel in the feature-map array by
# “how important this channel is” with regard to the “elephant” class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is the heatmap of the class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)
# For visualization purposes, you’ll also normalize the heatmap between 0 and 1
# Heatmap post-processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
# Superimposing the heatmap with the original picture
# Uses cv2 to load the original image
img = cv2.imread(img_path)
# Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# Converts the heatmap to RGB
heatmap = np.uint8(255 * heatmap)
# Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 0.4 here is a heatmap intensity factor.
superimposed_img = heatmap * 0.4 + img
# Saves the image to disk
cv2.imwrite('/Users/iliyanslavov/Downloads/elephant_cam.jpg', superimposed_img)
"""
The top three classes predicted for this image are as follows:
 African elephant (with 92.5% probability)
 Tusker (with 7% probability)
 Indian elephant (with 0.4% probability)
"""