import numpy as np
from typing import Union

# Функція втрат


class LossFunction:
    @staticmethod
    def calculate(y: np.array, y_hat: np.array) -> float:
        return -float(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


# Опис методів оптимізації


class GradientDescentHelper:
    def __init__(self, learning_rate: float):
        self.buffer = {}
        self.learning_rate = learning_rate

    def put(self, dW: np.array, db: np.array, layer_num: int) -> Union[None, bool]:
        self.buffer[layer_num] = {"dW": dW, "db": db}
        return True

    def pick(self, layer_num: int, key: str) -> np.array:
        return self.buffer[layer_num][key] * self.learning_rate


class MomentDescentHelper(GradientDescentHelper):
    def __init__(
        self, learning_rate: float, decay_rate: float = 0.9, corrected: bool = False
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta = decay_rate
        self.decay_buffer = {}
        self.corrected = corrected

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
            1 / (1 - self.beta ** (layer_num + 1)) if self.corrected else 1
        )
        return self.decay_buffer[layer_num][key] * self.learning_rate


class RMSPropHelper(MomentDescentHelper):
    def __init__(
        self,
        learning_rate: float,
        decay_rate: float = 0.99,
        corrected: bool = True,
        epsilon: float = 10e-5,
    ):
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
            1 / (1 - self.beta ** (layer_num + 1)) if self.corrected else 1
        )
        # print(self.decay_buffer)
        normed_derivative = current_derivative / (
            np.sqrt(self.decay_buffer[layer_num][key]) + self.eps
        )
        print(current_derivative)
        if np.sum(current_derivative) == np.inf:
            print("STOP")
        return normed_derivative * self.learning_rate
