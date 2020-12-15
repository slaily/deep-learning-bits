"""
Instantiating a model from an input tensor and a list of output tensors
"""
from keras import models
from model import model


# Extracts the outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layers_outputs)
