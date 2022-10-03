import numpy as np


# Таблиця спряженості
def confusion_matrix_binary(predicted: np.array, real: np.array) -> np.array:
    """
    Повертає таблицю спряженості для результатів бінарної класифікації
    :param predicted: Вектор прогнозів
    :param real: Вектор відгуків
    :return: Матриця спряженості
    """
    equals = predicted == real
    not_equal = predicted != real

    true_positive = np.sum(equals & (real == 0))
    true_negative = np.sum(equals & (real == 1))

    false_positive = np.sum(not_equal & (real == 1))
    false_negative = np.sum(not_equal & (real == 0))

    table_to_return = np.array(
        [[true_positive, false_negative], [false_positive, true_negative]]
    )

    return table_to_return


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
