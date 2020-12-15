import matplotlib.pyplot as plt


# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print('First activation layer: ', first_layer_activation)
print('First activation layer shape: ', first_layer_activation.shape)
# Visualizing the fourth channel
# Fourth channel of the activation of the first layer on the test cat picture
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# Visualizing the seventh channel
# Seventh channel of the activation of the first layer on the test cat picture
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
# At this point, let’s plot a complete visualization of all the activations in
# the network. You’ll extract and plot every channel in each of the eight activation
# maps, and you’ll stack the results in one big image tensor, with channels stacked side by side.
layer_names = []
# Names of the layers, so you can have them as part of your plot
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Displays the feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]
    # The feature map has shape (1, size, size, n_features).
    size = layer_activation.shape[1]
    # Tiles the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row & size))
    # Tiles each filter into a big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[
                0,
                :,
                :,
                col * images_per_row + row
            ]
            # Post-processes the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(
                figsize(
                    scale * display_grid.shape[1],
                    scale * display_grid.shape[0]
                )
            )
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
