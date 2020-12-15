"""
Visualizing convnet filters

Another easy way to inspect the filters learned by convnets is to display the visual pat- tern that each filter is meant to respond to. This can be done with gradient ascent in input space: applying gradient descent to the value of the input image of a convnet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

The process is simple: you’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then you’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. For instance, here’s a loss for the activation of filter 0 in the layer block3_conv1 of the VGG16 network, pretrained on ImageNet.
"""
import numpy as np

from keras.applications import VGG16
from keras import backend as K


model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])
# Obtaining the gradient of the loss with regard to the input
# The call to gradients returns a list of tensors (of size 1 in this case).
# Hence, you keep only the first element— which is a tensor.
grads = K.gradients(loss, model.input)[0]
# Gradient-normalization trick
# Add 1e–5 before dividing to avoid accidentally dividing by 0.
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# Fetching Numpy output values given Numpy input values
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
# Loss maximization via stochastic gradient descent
# Starts from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
# Magnitude of each gradient update
step = 1.

for i in range(40):
    # Computes the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])
    # Runs gradient ascent for 40 steps
    # Adjusts the input image in the direction that maximizes the loss
    input_img_data += grads_value * step

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    # Normalizes the tensor: centers on 0, ensures that std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # Clips to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # Converts to an RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


# Function to generate filter visualizations
# Builds a loss function that maximizes the activation of
# the nth filter of the layer under consideration
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Computes the gradient of the input picture with regard to this loss
    grads = K.gradients(loss, model.input)[0]
    # Normalization trick: normalizes the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # Returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    # Starts from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    # Runs gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]

    return decompress_image(img)


# Pattern that the zeroth channel in layer block3_conv1 responds to maximally
plt.imshow(generate_pattern('block3_conv', 0))
