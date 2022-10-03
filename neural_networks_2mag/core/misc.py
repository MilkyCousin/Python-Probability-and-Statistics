import numpy as np

# Генератори випадкових матриць


def basic_generator(num_row: int, num_col: int) -> np.array:
    """
    Функція, що генерує випадкову матрицю із компонентами з рівномірного на [0,1] розподілу.
    :param num_row: Кількість рядків вихідної матриці
    :param num_col: Кількість колонок вихідної матриці
    :return: Згенерована випадкова матриця
    """
    return np.random.randn(num_row, num_col)


def gaussian_generator(num_row: int, num_col: int) -> np.array:
    """
    Функція, що генерує випадкову матрицю із компонентами зі стандартного нормального розподілу.
    :param num_row: Кількість рядків вихідної матриці
    :param num_col: Кількість колонок вихідної матриці
    :return: Згенерована випадкова матриця
    """
    return np.random.normal(0, 1, (num_row, num_col))
