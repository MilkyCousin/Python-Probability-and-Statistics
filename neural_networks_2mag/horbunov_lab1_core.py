import matplotlib
import pickle as pkl
from datetime import datetime
from typing import Any, Callable, Dict, Union, List
from lab1.core.activations import *
from lab1.core.optimization import *
from lab1.core.misc import *

# Тому що дехто недбало поставив python
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Опис класу "Шар нейронної мережі"


class Layer:
    """
    Клас, що реалізує один шар щільної нейронної мережі.
    """

    def __init__(
        self,
        prev_num_neurons: int,
        num_neurons: int,
        generator: Callable,
        activation: ActivationFunction,
    ):
        """
        Ініціалізація класу, що реалізує шар мережі.
        :param prev_num_neurons: Кількість нейронів у попередньому шарі
        :param num_neurons: Кількість нейронів у новому шарі
        :param generator: Функція, що генерує випадкові матриці
        :param activation: Активаційна функція шару
        """
        self.prev_num_of_neurons = prev_num_neurons
        self.num_of_neurons = num_neurons
        self.activation = activation
        self.generator = generator

        self.W = self.generator(self.prev_num_of_neurons, self.num_of_neurons)
        self.b = np.zeros(shape=(1, self.num_of_neurons))

        self.Z_cur = None
        self.A_cur = None

    def forward(self, A_prev: np.array, training: bool) -> np.array:
        """
        Підрахунок значення активаційної функції на основі вхідних даних у шар.
        :param A_prev: Вхідні дані у поточний шар
        :param training: Чи тренується нейронна мережа?
        :return: Значення активаційної функції на основі вхідних даних
        """
        Z_cur = np.dot(A_prev, self.W) + self.b
        A_cur = self.activation.calculate(Z_cur)

        if training:
            self.Z_cur = Z_cur
            self.A_cur = A_cur

        return A_cur

    def backward(self) -> np.array:
        """
        Обчислення значення похідної активаційної функції
        :return: Значення похідної активаційної функції
        """
        return self.activation.calculate_derivative(self.Z_cur)

    def reset_params(self) -> None:
        """
        Зводить до 'заводських налаштувань' параметри мережі.
        :return: Нічого
        """
        self.W = self.generator(self.prev_num_of_neurons, self.num_of_neurons)
        self.b = self.generator(1, self.num_of_neurons)


# Опис класу "нейронна мережа"


class Net:
    """
    Клас, що описує щільну нейронну мережу.
    """

    def __init__(
        self,
        data_matrix: np.array,
        data_real: np.array,
        specification: Dict,
        optimizer_helper: GradientDescentHelper,
        generator: Callable = gaussian_generator,
        num_batches: int = 1,
    ):
        """
        Ініціалізація параметрів класу, що реалізує нейронну мережу.
        :param data_matrix: Матриця регресорів розмірності (n, p)
        :param data_real: Матриця відгуків розмірності (n, m)
        :param specification: Специфікація мережі: архітектура, кількість нейронів, активації на кожному шарі
        :param optimizer_helper: Допоміжна функція в оптимізації
        :param generator: Функція для генерування випадкових матриць
        :param num_batches: Кількість міні-пакетів для тренування (насправді вийде n або n + 1)
        """
        self.data = data_matrix
        self.real = data_real
        self.number_of_records = len(self.data)
        self.dimension = len(np.transpose(self.data))
        self.batch_size = self.number_of_records // num_batches
        self.layers_num = len(specification["layers_structure"])
        self.architecture = [self.dimension] + [
            current_spec["neurons"]
            for current_spec in specification["layers_structure"]
        ]
        self.layers = [
            Layer(
                prev_num_neurons=self.architecture[layer_num],
                num_neurons=self.architecture[layer_num + 1],
                activation=current_spec["activation"],
                generator=generator,
            )
            for layer_num, current_spec in enumerate(specification["layers_structure"])
        ]

        self.optimizer = optimizer_helper
        self.loss = LossFunction()

    def _forward(
        self,
        start: np.array = None,
        idx: Union[np.array, List] = None,
        training: bool = True,
    ) -> np.array:
        """
        Алгоритм forward propagation.
        :param start: Початкові дані, які треба загнати у вхідний шар
        :param idx: Набір номерів елементів, що треба відібрати з початкових даних
        :param training: Чи тренується наразі мережа?
        :return: Значення активаційної функції з останнього шару
        """
        A_prev = self.layers[0].forward(
            self.data[
                idx,
            ]
            if start is None
            else start,
            training=training,
        )

        for layer in self.layers[1:]:
            A_prev = layer.forward(A_prev, training=training)

        Y_prediction = A_prev

        return Y_prediction

    def pass_through(self, x: np.array) -> np.array:
        """
        Зробити прогноз нейронної мережі на основі вхідних даних x
        :param x: Матриця розмірності (n записів, p параметрів), на основі якої треба робити прогноз
        :return: Вектор прогнозів розмірності (n, n[L]), n[L] -- кількість нейронів у вихідному шарі мережі
        """
        return self._forward(start=x, idx=range(len(x)), training=False)

    def _backward(self, idx: Union[np.array, List] = None) -> Union[None, bool]:
        """
        Алгоритм зворотного розповсюдження похибки (backward propagation).
        :param idx: Набір номерів елементів, що треба відібрати з початкових даних
        :return: Якщо все правильно спрацювало, повертає 'True'
        """

        # Припускаємо, що працюємо з задачею біноміальної класифікації
        # Де L(y, y.hat) = sum(y * ln(y.hat) + (1-y) * ln(1-y.hat))
        # Та останній шар мережі має активаційну функцію у якості сигмоїди.
        # TODO: узагальнити, це ж не так важко :)
        assert self.layers[-1].activation == SigmoidFunction, "Invalid outer activation"
        m = len(idx)

        # Останній шар
        dZ = (
            self.layers[-1].A_cur
            - self.real[
                idx,
            ]
        )
        dW = np.dot(np.transpose(self.layers[-2].A_cur), dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Записуємо дані в оптимізатор
        self.optimizer.put(dW=dW, db=db, layer_num=len(self.layers) - 1)

        # Робимо крок по параметрам
        self.layers[-1].W -= self.optimizer.pick(len(self.layers) - 1, "dW") / m
        self.layers[-1].b -= self.optimizer.pick(len(self.layers) - 1, "db") / m

        # Інші шари у моделі
        for layer_num in list(reversed(range(self.layers_num)))[1:]:
            dZ = (
                np.dot(dZ, np.transpose(self.layers[layer_num + 1].W))
                * self.layers[layer_num].backward()
            )
            dW = np.dot(
                np.transpose(
                    self.layers[layer_num - 1].A_cur
                    if layer_num > 0
                    else self.data[
                        idx,
                    ]
                ),
                dZ,
            )
            db = np.sum(dZ, axis=0, keepdims=True)

            # Записуємо дані в оптимізатор
            self.optimizer.put(dW=dW, db=db, layer_num=layer_num)

            # Робимо крок по параметрам
            self.layers[layer_num].W -= self.optimizer.pick(layer_num, "dW") / m
            self.layers[layer_num].b -= self.optimizer.pick(layer_num, "db") / m

        return True

    def train_one_epoch(self) -> float:
        """
        Тренуємо мережу за одну епоху. Тобто, спочатку виконуємо алгоритм forward propagation.
        Далі, підрахунок градієнтів та оновлення параметрів у backward propagation.
        Тренування проходить міні-пакетами.
        :return: Вибіркове середнє втрат за всіма міні-пакетами
        """
        number_of_packed_batches = self.number_of_records // self.batch_size
        residual = self.number_of_records - self.batch_size * number_of_packed_batches
        loss = []

        for k in range(number_of_packed_batches):
            idx = np.arange(0, self.batch_size, 1) + self.batch_size * k
            current_prediction = self._forward(idx=idx)
            current_loss = self.loss.calculate(
                y=self.real[
                    idx,
                ],
                y_hat=current_prediction,
            )
            loss.append(current_loss)
            self._backward(idx=idx)

        if residual > 0:
            idx = np.arange(0, residual, 1) + self.batch_size * number_of_packed_batches
            current_prediction = self._forward(idx=idx)
            current_loss = self.loss.calculate(
                y=self.real[
                    idx,
                ],
                y_hat=current_prediction,
            )
            loss.append(current_loss)
            self._backward(idx=idx)

        mean_loss = float(np.mean(loss))
        return mean_loss

    def train_model(self, num_epochs: int, printable=True) -> Union[None, bool]:
        """
        Тренування моделі за скінченну кількість епох.
        :param num_epochs: Кількість епох тренування мережі
        :param printable: Чи виводити додаткову інформацію під час тренування?
        :return: Якщо все правильно спрацювало, повертає 'True'
        """
        num_digits = len(str(num_epochs))

        for epoch_num in range(num_epochs):
            current_loss = self.train_one_epoch()
            if printable and (epoch_num + 1) % 1000 == 0:
                print(
                    f"Epoch #{str(epoch_num + 1).ljust(int(num_digits))}:\tLoss = {current_loss}"
                )
        return True

    def reset_params(self) -> Union[None, bool]:
        """
        Зводить до 'заводських налаштувань' параметри мережі.
        :return: Якщо все правильно спрацювало, повертає 'True'
        """
        for layer_num in range(len(self.layers)):
            self.layers[layer_num].reset_params()
        return True

    def save(self, name) -> None:
        """
        Консервує параметри мережі для подальшого можливого зчитування
        :param name: Шлях, назва файлу, в якому зберігатиметься інформація про параметри мережі
        :return: Нічого
        """
        with open(f"./{name}.pkl", "wb") as out:
            out_dict = {"architecture": self.architecture, "layers": {}}
            for layer_num, layer in enumerate(self.layers):
                out_dict["layers"][layer_num] = {"W": layer.W, "b": layer.b}
            pkl.dump(out_dict, out)


# Візуалізація роботи класифікатора


def decision_boundary(
    classifier: Any,
    x: np.array,
    y: np.array,
    h: float = 0.05,
    fig_size: tuple = (10, 10),
    name: str = str(datetime.today()).strip(" "),
) -> None:
    """
    Малює множини рішень на основі взятого класифікатора. Утворений малюнок зберігається у .png файлі.
    :param classifier: Класифікатор
    :param x: Двовимірна матриця регресорів
    :param y: Вектор відгуків
    :param h: Параметр щільності сітки для побудови контурного графіка
    :param fig_size: Розмір вихідної картинки
    :param name: Шлях, назва файлу, в якому зберігатиметься картинка
    :return: Нічого
    """
    fig, ax = plt.subplots(figsize=fig_size)

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = classifier(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    plt.savefig(f"./pics/{name}.png")
    plt.clf()


# Класифікація на основі натренованої мережі


def nn_classifier(nn_unit: Net, p: float = 0.5) -> Callable:
    """
    На основі нейронної мережі для задачі бінарної класифікації, повернути рішуче правило.
    :param nn_unit: Сутність класу Net / Нейронна мережа
    :param p: Параметр правдоподібності
    :return: Рішуче правило на основі взятої мережі
    """

    def to_return(x: np.array):
        """
        Рішуче правило бінарної класифікації
        :param x: Матриця розмірності (n - записів, p - параметрів)
        :return: Прогноз номерів класів у вигляді вектора розмірності (n, 1)
        """
        return np.squeeze(np.array(nn_unit.pass_through(x) > p, dtype=int))

    return to_return


if __name__ == "__main__":
    from lab1.core.activations import ReLuFunction, SigmoidFunction
    import numpy as np

    structure = {
        "layers_structure": [
            {"neurons": 3, "activation": ReLuFunction},
            {"neurons": 6, "activation": ReLuFunction},
            {"neurons": 1, "activation": SigmoidFunction},
        ],
    }

    np.random.seed(123809123)
    n = 100
    t = np.arange(0, 1, 1 / n)
    T = np.hstack(
        (np.random.normal(0, 0.02, n // 2), np.random.normal(2, 0.03, n // 2))
    )
    X = np.transpose(np.vstack((t, T)))
    Y = np.hstack((np.zeros(n // 2), np.ones(n // 2))).reshape((n, 1))

    nn = Net(
        data_matrix=X,
        data_real=Y,
        specification=structure,
        optimizer_helper=GradientDescentHelper(learning_rate=0.01),
    )

    nn.train_model(100)

    def nn_classifier(x, p=0.5):
        return np.squeeze(np.array(nn.pass_through(x) > p, dtype=int))

    decision_boundary(classifier=nn_classifier, x=X, y=Y, h=0.01, fig_size=(10, 10))
