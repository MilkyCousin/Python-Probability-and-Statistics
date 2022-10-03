import numpy as np

# Активаційні функції


class ActivationFunction:
    """
    Базовий клас, що описує певну активаційну функцію нейронної мережі.
    Має два методи: прямий підрахунок активації та обчислення значення похідної активації.
    """

    @staticmethod
    def calculate(x: np.array) -> np.array:
        """
        Безпосередній підрахунок активаційної функції.
        :param x: Матриця значень, в яких треба обчислити активацію
        :return: Матриця такої ж розмірності що x зі значень активації
        """
        return x

    @staticmethod
    def calculate_derivative(x: np.array) -> np.array:
        """
        Безпосередній підрахунок похідної активаційної функції.
        :param x: Матриця значень, в яких треба обчислити похідну активації
        :return: Матриця такої ж розмірності що x зі значень похідної активації
        """
        return 1 + 0 * x


class SigmoidFunction(ActivationFunction):
    @staticmethod
    def calculate(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def calculate_derivative(x: np.array) -> np.array:
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)


class ReLuFunction(ActivationFunction):
    @staticmethod
    def calculate(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def calculate_derivative(x: np.array) -> np.array:
        return 1 * (x >= 0)


class TanhFunction(ActivationFunction):
    @staticmethod
    def calculate(x: np.array) -> np.array:
        return np.tanh(x)

    @staticmethod
    def calculate_derivative(x: np.array) -> np.array:
        sum_of_exp = np.exp(x) + np.exp(-x)
        return 4 / (sum_of_exp * sum_of_exp)
