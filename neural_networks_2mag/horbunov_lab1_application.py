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
        {"neurons": 2, "activation": TanhFunction},
        {"neurons": 1, "activation": SigmoidFunction},
    ],
}
num_epochs = 10000
num_of_batches = 20

print("=== Classic method ===")
# Класичний градієнтний спуск
gd_helper = GradientDescentHelper(learning_rate=0.005)
nn_classic = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=gd_helper,
    num_batches=num_of_batches,
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
print("=== Moment method ===")
# Моментний градієнтний спуск
mgd_helper = MomentDescentHelper(learning_rate=0.005)
nn_moment = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=mgd_helper,
    num_batches=num_of_batches,
)
nn_moment.train_model(num_epochs=num_epochs)

nn_moment_classifier = nn_classifier(nn_moment)
current_name = "moment" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_moment_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)
print("=== RMS method ===")
# RMS-propagation
rms_helper = RMSPropHelper(learning_rate=0.001)
nn_rms = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=rms_helper,
    num_batches=num_of_batches,
)
nn_rms.train_model(num_epochs=num_epochs)

nn_rms_classifier = nn_classifier(nn_rms)
current_name = "rms" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_rms_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)
print("=== Adam method ===")
# Adam
adam_helper = AdamHelper(
    learning_rate=0.001, moment_decay_rate=0.9, rms_decay_rate=0.999
)
nn_adam = Net(
    data_matrix=X,
    data_real=Y,
    specification=given_specification,
    optimizer_helper=adam_helper,
    num_batches=num_of_batches,
)
nn_adam.train_model(num_epochs=num_epochs)

nn_adam_classifier = nn_classifier(nn_adam)
current_name = "adam" + str(datetime.today()).strip(" ")
decision_boundary(
    classifier=nn_adam_classifier,
    x=X,
    y=Y,
    h=0.01,
    fig_size=(10, 10),
    name=current_name,
)
