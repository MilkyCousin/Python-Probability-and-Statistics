import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from lab1.core.activations import *
from lab1.core.optimization import *
from lab1.core.misc import confusion_matrix_binary
from horbunov_lab1_core import Net, decision_boundary, nn_classifier

# Зчитуємо дані
target_path = r"./data4.csv"
col_names = ["X1", "X2", "Y"]
given_data = pd.read_csv(target_path, sep=" ", names=col_names, header=None)
# Записуємо у відповідні матриці
X = np.array(given_data[["X1", "X2"]])
Y = np.array(given_data[["Y"]])
# Кількість спостережень
n = len(X)
# Кількість спостережень у тренувальній вибірці
n_train = int(n * 0.85)
# Відбираємо номери спостережень у тренувальну вибірку
np.random.seed(56785)
idx = np.random.choice(np.arange(0, n), size=n_train, replace=False)
# Створюємо тренувальну вибірку
X_train = X[idx, ]
Y_train = Y[idx, ]
# Створюємо тестову вибірку
X_test = X[np.delete(np.arange(0, n), idx), ]
Y_test = Y[np.delete(np.arange(0, n), idx), ]
# Будуємо нейронну мережу
given_specification = {
    "layers_structure": [
        {"neurons": 4, "activation": TanhFunction},
        {"neurons": 3, "activation": TanhFunction},
        {"neurons": 2, "activation": TanhFunction},
        {"neurons": 1, "activation": SigmoidFunction},
    ],
}
num_epochs = 10000
num_of_batches = 20
# Лише для відтворення результатів
np.random.seed(1)
# Класичний градієнтний спуск
rms_helper = RMSPropHelper(learning_rate=0.01, decay_rate=0.99)
nn_rms = Net(
    data_matrix=X_train,
    data_real=Y_train,
    specification=given_specification,
    optimizer_helper=rms_helper,
    num_batches=num_of_batches,
)
nn_rms.train_model(num_epochs=num_epochs)

nn_rms_classifier = nn_classifier(nn_rms)
current_name = "classic" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_rms_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)

print("Таблиця спряженості на тренувальних даних:")
Y_prediction_train = nn_rms.pass_through(X_train)
print(confusion_matrix_binary((Y_prediction_train > 0.5) * 1, Y_train))

print("Таблиця спряженості на тестових даних:")
Y_prediction_test = nn_rms.pass_through(X_test)
print(confusion_matrix_binary((Y_prediction_test > 0.5) * 1, Y_test))
