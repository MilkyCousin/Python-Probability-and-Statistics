import numpy as np
from typing import Dict

# Функція втрат


class LossFunction:
    @staticmethod
    def calculate(y, y_hat) -> float:
        return -float(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


# Опис методів оптимізації


class GradientDescentHelper:
    def __init__(self, learning_rate: float):
        self.buffer = {}
        self.learning_rate = learning_rate

    def put(self, dW, db, layer_num):
        self.buffer[layer_num] = {"dW": dW, "db": db}
        return True

    def pick(self, layer_num, key):
        assert key in ["dW", "db"], "Invalid key entered"
        return self.buffer[layer_num][key] * self.learning_rate


class MomentDescentHelper(GradientDescentHelper):
    def __init__(self, learning_rate: float, decay_coefficient: float=1):
        super().__init__(learning_rate=learning_rate)
        self.decay_coefficient = decay_coefficient

    def put(self, dW, db, layer_num):
        self.buffer[layer_num] = {"dW": dW, "db": db, "dW_dec": 0, "db_dec": 0}
        return True

    def pick(self, layer_num, key):
        assert key in ["dW", "db"], "Invalid key entered"
        deriv = self.decaying[key].next(self.buffer[key])
        return self.buffer[layer_num][key] * self.learning_rate
