import os
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, accuracy_score

np.random.seed(0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.where(x >= 0, x, 0)

def d_relu(x):
    return np.where(x >= 0, 1, 0)

def leaky_relu(x, lp=0.01):
    return np.where(x >= 0, x, lp*x)

def d_leaky_relu(x, lp=0.01):
    return np.where(x >= 0, 1, lp)

def d_tanh(x):
    return 4/(np.exp(x) + np.exp(-x))

def entropy(u, v):
    cost = (-1) * (1 / v.shape[0]) * np.sum(v.T @ np.log(u) + (1-v).T @ np.log(1-u))
    return np.squeeze(cost)


class ExceptionNN(Exception):
    pass


class ClassLayerNN:

    def __init__(self, input_number, output_number, activation, activation_derivative):
        self._activation = activation
        self._activation_derivative = activation_derivative

        self._a = None

        self._z = None
        self._dz = None

        #self._w = np.random.normal(size=(input_number, output_number))
        self._w = np.random.randn(input_number, output_number)
        self._dw = None

        self._b = np.zeros(shape=(1, output_number))
        self._db = None

    def input(self, x):
        self._z = x @ self._w + self._b
        self._a = self._activation(self._z)

    def gradient(self, x, a, m):
        self._dz = x
        self._dw = (a.T @ self._dz)/m
        self._db = np.sum(self._dz, axis=0, keepdims=True)/m

    def derivative(self, x):
        return self._activation_derivative(x)

    def update(self, lr):
        if any(u is None for u in [self._db, self._dw, self._dz]):
            raise ExceptionNN("Gradients weren't initialised")
        self._w = self._w - lr * self._dw
        self._b = self._b - lr * self._db

    def __str__(self):
        return f'shapes: b[{self._b.shape}], w[{self._w.shape}]'

    @property
    def return_gradient(self):
        return self._dz

    @property
    def return_weights(self):
        return self._w

    @property
    def return_inner(self):
        return self._z

    @property
    def return_outer(self):
        return self._a


class SigmoidLayerNN(ClassLayerNN):

    def __init__(self, input_number, output_number):
        ClassLayerNN.__init__(self, input_number, output_number,
                              activation=sigmoid,
                              activation_derivative=d_sigmoid)


class RELULayerNN(ClassLayerNN):

    def __init__(self, input_number, output_number):
        ClassLayerNN.__init__(self, input_number, output_number,
                              activation=relu,
                              activation_derivative=d_relu)


class LeakyRELULayerNN(ClassLayerNN):

    def __init__(self, input_number, output_number, lp=0.01):
        ClassLayerNN.__init__(self, input_number, output_number,
                              activation=lambda u: leaky_relu(u, lp),
                              activation_derivative=lambda u: d_leaky_relu(u, lp))


class TanhLayerNN(ClassLayerNN):

    def __init__(self, input_number, output_number):
        ClassLayerNN.__init__(self, input_number, output_number,
                              activation=np.tanh,
                              activation_derivative=d_tanh)


class ClassNN:

    def __init__(self, inner_layers_class, outer_layers_class,
                 num_neurons_list, lr=0.01, max_iter=100, cost_fun=entropy):

        self._L = len(num_neurons_list)-1
        self._l = np.empty(self._L, dtype=object)

        for i in range(1, self._L):
            self._l[i-1] = inner_layers_class(num_neurons_list[i-1], num_neurons_list[i])

        self._l[self._L-1] = outer_layers_class(num_neurons_list[self._L-1], num_neurons_list[self._L])
        self._cost_function = cost_fun

        self._lr = lr
        self._mx_iter = max_iter

    def _forward(self, x):
        self._l[0].input(x)
        for li in range(1, self._L):
            self._l[li].input(self._l[li-1].return_outer)
        return self._l[self._L-1].return_outer

    def _backward(self, x, y):
        m = x.shape[0]
        self._l[self._L-1].gradient(self._l[self._L-1].return_outer - y,
                                    self._l[self._L-2].return_outer, m)

        for li in range(self._L-1, 1, -1):
            dz = (self._l[li].return_gradient @ self._l[li].return_weights.T) * \
                 self._l[li-1].derivative(self._l[li-1].return_outer)
            self._l[li-1].gradient(dz, self._l[li-2].return_outer, m)

        dz = (self._l[1].return_gradient @ self._l[1].return_weights.T) * \
             self._l[0].derivative(self._l[0].return_outer)
        self._l[0].gradient(dz, x, m)

    def _update(self):
        for li in range(0, self._L):
            self._l[li].update(self._lr)

    def train(self, x, y, bp=0, sp=1, dp=0, ds=100, dn=''):
        for i in range(self._mx_iter):
            yt = self._forward(x)
            self._backward(x, y)

            self._update()
            cost = self._cost_function(yt, y)
            rmse = (mean_squared_error(y, yt)) ** 0.5

            if bp and not i % sp:
                print('Iteration #%i' % (i+1), 'Cost: %.4f' % cost, sep='\t')
                print('Iteration #%i' % (i+1), 'RMSE: %.4f' % rmse, sep='\t')
            if dp and not i % ds:
                self.decision_boundary(x, y, name=dn + str(i + 1))

    def predict(self, x, p=0.5):
        return np.squeeze(np.array(self._forward(x) > p, dtype=int))

    def raw_predict(self, x):
        return self._forward(x)

    def pickle(self, name, path=''):
        with open(path if path else os.path.join(os.getcwd(), name + '.pkl'), 'wb') as f:
            pkl.dump(self, f)

    # one student advised to use it for boundaries plotting
    def decision_boundary(self, x, y, name: str = '', ax: plt.axis = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        h = 0.02

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        print(np.c_[xx.ravel(), yy.ravel()].shape)
        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        ax.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if name:
            plt.savefig(os.path.join(os.getcwd(), name + '.png'))
            plt.clf()
        else:
            plt.show()
            plt.clf()

    def score(self, y_true, y_predicted):
        return accuracy_score(y_true, y_predicted)


if __name__ == '__main__':
    X_y = pd.read_csv('./data1.csv', sep=',')
    X = X_y[['x1', 'x2']].to_numpy()
    Y = X_y['y']
    M1 = Y == 1
    Y = np.array(Y).reshape(len(Y), 1)

    sb.scatterplot(X[:,0][M1], X[:,1][M1], color='orange')
    sb.scatterplot(X[:,0][~M1], X[:,1][~M1], color='purple')
    plt.savefig('scatter.png')

    import glob
    for u in glob.glob('~/PycharmProjects/osvm/ml/nn/animation/'):
        os.remove(u)

    l_num = [X.shape[1], 9, 5, 1]
    nn = ClassNN(TanhLayerNN, SigmoidLayerNN, l_num, lr=0.01, max_iter=8350)
    nn.train(X, Y, bp=1, sp=100, dp=1, ds=150, dn='./animation/train_pic')

    predicted = nn.predict(X)

    print(classification_report(Y, predicted))
    print(nn.score(Y, predicted))
    nn.decision_boundary(X, Y)
    nn.decision_boundary(X, Y, name='nn_result')

    nn.pickle('nn_result')

    os.system(
        'convert -delay 5 -loop 0 ~/PycharmProjects/osvm/ml/nn/animation/train*.png ~/PycharmProjects/osvm/ml/nn/animation/result.mp4'
    )
