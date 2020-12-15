"""
Generating a grid of all filter response patterns in a layer
"""
layer_name = 'block1_conv'
size = 64
margin = 5
# Empty (black) image to store results
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

# Iterates over the rows of the results grid
for i in range(8):
    # Iterates over the columns of the results grid
    for j in range(8):
        # Generates the pattern for filter i + (j * 8) in layer_name
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        # Puts the result in the square (i, j) of the results grid
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[
            horizontal_start: horizontal_end,
            vertical_start: vertical_end,
            :
        ]

# Displays the results grid
plt.figure(figsize=(20, 20))
plt.imshow(results)