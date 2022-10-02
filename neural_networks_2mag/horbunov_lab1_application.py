import pandas as pd
from datetime import datetime
from lab1.core.activations import *
from lab1.core.optimization import *
from horbunov_lab1_core import Net, decision_boundary, nn_classifier


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
num_epochs = 10000

# Класичний градієнтний спуск
gd_helper = GradientDescentHelper(learning_rate=0.005)
nn_classic = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=gd_helper,
    num_batches=1,
)
nn_classic.train_model(num_epochs=num_epochs)

nn_classic_classifier = nn_classifier(nn_classic)
current_name = "classic" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_classic_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)

# Моментний градієнтний спуск
mgd_helper = MomentDescentHelper(learning_rate=0.005, decay_rate=0.9)
nn_moment = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=mgd_helper,
    num_batches=1,
)
nn_moment.train_model(num_epochs=num_epochs)

nn_moment_classifier = nn_classifier(nn_classic)
current_name = "moment" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_moment_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)
