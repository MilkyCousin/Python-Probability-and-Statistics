import pandas as pd
from lab1.core.activations import *
from lab1.core.optimization import *
from horbunov_lab1_core import Net, decision_boundary


target_path = r"./data4.csv"
col_names = ["X1", "X2", "Y"]
given_data = pd.read_csv(target_path, sep=" ", names=col_names, header=None)

X = np.array(given_data[["X1", "X2"]])
Y = np.array(given_data[["Y"]])

given_specification = {
    "layers_structure": [
        {"neurons": 4, "activation": TanhFunction},
        {"neurons": 6, "activation": TanhFunction},
        {"neurons": 4, "activation": TanhFunction},
        {"neurons": 4, "activation": TanhFunction},
        {"neurons": 1, "activation": SigmoidFunction},
    ],
}
gd_helper = GradientDescentHelper(learning_rate=0.001)
nn_classic = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=gd_helper,
    num_batches=1,
)

num_epochs = 10000
nn_classic.train_model(num_epochs=num_epochs)


def nn_classifier(x, p=0.5):
    return np.squeeze(np.array(nn_classic.pass_through(x) > p, dtype=int))


decision_boundary(
    classifier=nn_classifier, x=X, y=Y, h=0.05, fig_size=(10, 10)
)

nn_classic.save("nn_classic")
