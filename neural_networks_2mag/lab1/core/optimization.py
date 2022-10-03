import numpy as np
from typing import Union

# Функція втрат


class LossFunction:
    """
    Функція втрат в задачі бінарної класифікації.
    Має метод (далі матиме два): прямий підрахунок втрат (та підрахунок похідної).
    """

    @staticmethod
    def calculate(y: np.array, y_hat: np.array) -> float:
        """
        Безпосередній підрахунок функції втрат, маючи вектор відгуків та векторів прогнозів.
        Повертає значення функції втрат.
        :param y: Вектор відгуків
        :param y_hat: Векторів прогнозів
        :return: Значення функції втрат
        """
        return -float(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


# Опис методів оптимізації
# Класи нижче -- це один із можливих способів реалізувати той чи інший метод оптимізації.
# Реалізація, що наведена нижче, базується на модифікації вже порахованих градієнтів.
# Тобто на вході передаємо дані про градієнти, на виході -- крокові значення для параметрів.


class GradientDescentHelper:
    """
    Допоміжний клас для підрахунку крокового значення в методі градієнтного спуску.
    """

    def __init__(self, learning_rate: float):
        """
        У конструкторі визначаємо змінну, яка зберігатиме інформацію про поточні градієнти: buffer.
        Також зберігаємо у learning_rate вказане значення параметру навчання моделі.
        :param learning_rate: Параметр навчання моделі
        """
        self.buffer = {}
        self.learning_rate = learning_rate

    def put(self, dW: np.array, db: np.array, layer_num: int) -> Union[None, bool]:
        """
        Метод для запису поточних значень градієнтів dW, db деякого (layer_num)-го шару.
        :param dW: Градієнт d(Loss)/dW[l], де Loss -- функція втрат
        :param db: Градієнт d(Loss)/db[l]
        :param layer_num: layer_num = l -- номер шару в мережі
        :return: Якщо все правильно спрацьовано, то повертає 'True'
        """
        self.buffer[layer_num] = {"dW": dW, "db": db}
        return True

    def pick(self, layer_num: int, key: str) -> np.array:
        """
        Повернути значення кроків оновлення для параметрів відповідного шару в мережі
        :param layer_num: Номер шару в мережі
        :param key: Який саме брати градієнт, за матрицею W[l] чи за вектором b[l]?
        :return: Крокове значення для оновлення відповідного параметра в мережі
        """
        return self.buffer[layer_num][key] * self.learning_rate


class MomentDescentHelper(GradientDescentHelper):
    """
    Допоміжний клас для підрахунку крокового значення в методі моментного градієнтного спуску.
    """

    def __init__(
        self, learning_rate: float, decay_rate: float = 0.9, corrected: bool = False
    ):
        """
        У конструкторі визначаємо змінну, яка зберігатиме інформацію про поточні градієнти: buffer.
        Також зберігаємо у learning_rate вказане значення параметру навчання моделі.
        :param learning_rate: Параметр навчання моделі
        :param decay_rate: Параметр затухання в експоненційно зважених середніх
        :param corrected: Чи робити поправку на зміщення відповідних зважених середніх?
        """
        super().__init__(learning_rate=learning_rate)
        self.beta = decay_rate
        self.decay_buffer = {}
        self.corrected = corrected
        self.step = 1

    def put(self, dW: np.array, db: np.array, layer_num: int) -> Union[None, bool]:
        self.buffer[layer_num] = {"dW": dW, "db": db}
        if layer_num not in self.decay_buffer:
            self.decay_buffer[layer_num] = {"dW": 0, "db": 0}
        return True

    def pick(self, layer_num: int, key: str) -> np.array:
        current_derivative = self.buffer[layer_num][key]
        previous_decay = self.decay_buffer[layer_num][key]
        next_decay = self.beta * previous_decay + (1 - self.beta) * current_derivative
        self.decay_buffer[layer_num][key] = next_decay * (
            1 / (1 - self.beta**self.step) if self.corrected else 1
        )
        self.step += 1
        return self.decay_buffer[layer_num][key] * self.learning_rate


class RMSPropHelper(MomentDescentHelper):
    """
    Допоміжний клас для підрахунку крокового значення в методі Root Mean Square Propagation.
    """

    def __init__(
        self,
        learning_rate: float,
        decay_rate: float = 0.99,
        corrected: bool = True,
        epsilon: float = 10e-7,
    ):
        """
        У конструкторі визначаємо змінну, яка зберігатиме інформацію про поточні градієнти: buffer.
        Також зберігаємо у learning_rate вказане значення параметру навчання моделі.
        :param learning_rate: Параметр навчання моделі
        :param decay_rate: Параметр затухання в експоненційно зважених середніх
        :param corrected: Чи робити поправку на зміщення відповідних зважених середніх?
        :param epsilon: Коригувальний доданок у знаменнику для арифметичної стійкості
        """
        MomentDescentHelper.__init__(
            self,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            corrected=corrected,
        )
        self.eps = epsilon

    def pick(self, layer_num: int, key: str) -> np.array:
        current_derivative = self.buffer[layer_num][key]
        current_decay = self.decay_buffer[layer_num][key]
        next_decay = self.beta * current_decay + (1 - self.beta) * (
            current_derivative * current_derivative
        )
        self.decay_buffer[layer_num][key] = next_decay * (
            1 / (1 - self.beta**self.step) if self.corrected else 1
        )
        normed_derivative = current_derivative / (
            np.sqrt(self.decay_buffer[layer_num][key]) + self.eps
        )
        self.step += 1
        return normed_derivative * self.learning_rate


class AdamHelper(GradientDescentHelper):
    """
    Допоміжний клас для підрахунку крокового значення в методі Adaptive Moment Estimation.
    """

    def __init__(
        self,
        learning_rate: float,
        moment_decay_rate: float,
        rms_decay_rate: float,
        epsilon: float = 10e-7,
    ):
        """
        У конструкторі визначаємо змінну, яка зберігатиме інформацію про поточні градієнти: buffer.
        Також зберігаємо у learning_rate вказане значення параметру навчання моделі.
        :param learning_rate: Параметр навчання моделі
        :param moment_decay_rate: Параметр затухання в експоненційно зважених середніх у моментному методі
        :param rms_decay_rate: Параметр затухання в експоненційно зважених середніх у методі RMSProp
        :param epsilon: Коригувальний доданок у знаменнику для арифметичної стійкості
        """
        super().__init__(learning_rate)
        self.buffer_moment = {}
        self.buffer_rms = {}
        self.buffer = {}

        self.alpha = learning_rate

        self.decay_moment = moment_decay_rate
        self.decay_rms = rms_decay_rate

        self.eps = epsilon
        self.step = 1

    def put(self, dW: np.array, db: np.array, layer_num: int) -> Union[None, bool]:
        self.buffer[layer_num] = {"dW": dW, "db": db}

        if layer_num not in self.buffer_moment:
            self.buffer_moment[layer_num] = {"dW": 0, "db": 0}
        if layer_num not in self.buffer_rms:
            self.buffer_rms[layer_num] = {"dW": 0, "db": 0}

        return True

    def pick(self, layer_num: int, key: str) -> np.array:
        current_derivative = self.buffer[layer_num][key]

        previous_decay_moment = self.buffer_moment[layer_num][key]
        previous_decay_rms = self.buffer_rms[layer_num][key]

        next_decay_moment = (
            self.decay_moment * previous_decay_moment
            + (1 - self.decay_moment) * current_derivative
        )
        self.buffer_moment[layer_num][key] = next_decay_moment / (
            1 - self.decay_moment**self.step
        )

        next_decay_rms = self.decay_rms * previous_decay_rms + (1 - self.decay_rms) * (
            current_derivative * current_derivative
        )
        self.buffer_rms[layer_num][key] = next_decay_rms / (
            1 - self.decay_rms**self.step
        )

        self.step += 1
        return self.alpha * (
            self.buffer_moment[layer_num][key]
            / (np.sqrt(self.buffer_rms[layer_num][key]) + self.eps)
        )
