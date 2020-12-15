import numpy as np


def naive_relu(x):
    # X must be 2D Numpy tensor
    assert len(x.shape) == 2
    # Avoid overwriting the input tensor
    x = x.copy()

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            x[row, col] = max(x[row, col], 0)

    return x


def naive_add(x, y):
    # X must be 2D Numpy tensor
    assert len(x.shape) == 2
    assert x.shape == y.shape
    # Avoid overwriting the input tensor
    x = x.copy()

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            x[row, col] += y[row, col]

    return x


def naive_add_matrix_and_vector(x, y):
    # X must be 2D Numpy tensor
    assert len(x.shape) == 2
    # Y must be Numpy vector
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    # Avoid overwriting the input tensor
    x = x.copy()

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            x[row, col] = y[col]

    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.

    for row in range(x.shape[0]):
        z += x[row] * y[row]

    return z


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            z[row] += x[row, col] * y[col]

    return z


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))

    for row in range(x.shape[0]):
        for col in range(y.shape[1]):
            row_x = x[row, :]
            column_y = y[:, col]
            z[row, col] = naive_vector_dot(row_x, column_y)

    return z


# 2D Numpy tensor
print('*** Naive ReLU ***')
input_x_2d_tensor = np.array(
    [
        [5, 78, 2, 34, 0],
        [6, 79, 3, 35, 1],
        [7, 80, 4, 36, 2]
    ]
)
print('Input 2D tensor shape: ', input_x_2d_tensor.shape)
print('Input 2D tensor ndim: ', input_x_2d_tensor.ndim)
output_2d_tensor = naive_relu(input_x_2d_tensor)
print('ReLU output: ', output_2d_tensor)
print('*** ---------- ***')
print()
print('*** Naive ADD operation ***')
input_y_2d_tensor = np.array(
    [
        [1, 7, 22, 31, 0],
        [3, 9, 33, 35, 1],
        [4, 8, 44, 36, 2]
    ]
)
output_2d_tensor = naive_add(input_x_2d_tensor, input_y_2d_tensor)
print('Naive ADD output: ', output_2d_tensor)
print('*** ---------- ***')
print()
print('*** Naive ADD operation - Matrix and Vector ***')
input_x_2d_tensor = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
)
input_y_2d_tensor = np.array([9, 10, 11, 12])
output_2d_tensor = naive_add_matrix_and_vector(input_x_2d_tensor, input_y_2d_tensor)
print('Naive ADD operation - Matrix and Vector output: ', output_2d_tensor)
print('*** ---------- ***')
print()
print('*** Naive DOT operation - Vector ***')
x_vector = np.array([1, 2, 3])
y_vector = np.array([4, 5, 6])
dot_product = naive_vector_dot(x_vector, y_vector)
print('Naive DOT output: ', dot_product)
print('*** ---------- ***')
print()
print('*** Naive DOT operation - Matrix and Vector ***')
x_matrix = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
)
y_vector = np.array([1, 2, 1, 2])
dot_product = naive_matrix_vector_dot(x_matrix, y_vector)
print('Naive DOT operation - Matrix and Vector output: ', dot_product)
print('*** ---------- ***')
print()
print('*** Naive DOT operation - Matrix ***')
x_matrix = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
)
y_matrix = np.array(
    [
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0]
    ]
)
dot_product = naive_matrix_dot(x_matrix, y_matrix)
print('Naive DOT operation - Matrix: ', dot_product)
print('*** ---------- ***')
print()
