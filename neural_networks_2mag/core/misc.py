import numpy as np

# Генератори випадкових матриць


def basic_generator(num_row: int, num_col: int):
    return np.random.randn(num_row, num_col)


def gaussian_generator(num_row: int, num_col: int):
    return np.random.normal(0, 1, (num_row, num_col))
