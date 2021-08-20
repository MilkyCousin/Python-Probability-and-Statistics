import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error


DIGITS = 4
MSG = """Estimated RMSE: {val}"""

seed = 0
np.random.seed(seed)


class PMF:

    def __init__(self, d_param: int = 1, max_iter_number: int = 10, debugging: bool = True, step: int = 1):

        self._V = None
        self._U = None

        self._N1 = 0
        self._N2 = 0

        self._d = d_param
        self._iter_num = max_iter_number

        self._parameters = dict()
        self._cost = 0

        self._score = 0

        self._debug = debugging
        self._step = step

    def _log_likelihood(self, x: np.ndarray, i: np.ndarray, lr: float, sigma: float):
        value = (1/(sigma ** 2)) * np.linalg.norm(
            i * (x - np.dot(self._U.T, self._V)), ord='fro') ** 2 + lr * (
                np.linalg.norm(self._U, ord='fro') ** 2 + np.linalg.norm(self._V, ord='fro') ** 2
        )
        return np.log(.5 * value)

    def _pmf(self, x: np.ndarray, lr: float, sigma: float, eps: float):

        indicator = np.array(x != 0, dtype=int).reshape(x.shape)

        for iteration in range(self._iter_num):

            for i in range(self._N1):
                self._U[:, i] = ((1/(lr * (sigma ** 2) + np.sum(
                    [
                        indicator[i, j] * np.linalg.norm(self._V[:, j].reshape((self._d, 1)), ord='fro') ** 2
                        for j in range(self._N2)
                    ]
                ))) * np.dot(self._V, (indicator * x)[i, ].reshape((1, self._N2)).T)).reshape((self._d,))

            for j in range(self._N2):
                self._V[:, j] = ((1/(lr * (sigma ** 2) + np.sum(
                    [
                        indicator[i, j] * np.linalg.norm(self._U[:, i].reshape((self._d, 1)), ord='fro') ** 2
                        for i in range(self._N1)
                    ]
                ))) * np.dot(self._U, (indicator * x)[:, j].reshape((1, self._N1)).T)).reshape((self._d,))

            cost = self._log_likelihood(x, indicator, lr, sigma)
            delta = np.abs(cost - self._cost)
            self._cost = cost

            if self._debug and not iteration % self._step:
                print(iteration+1, np.round(self._cost, DIGITS), np.round(delta, DIGITS), sep='\t', end='\n')

            self._parameters.update({'sigma': sigma, 'lambda': lr, 'dim': self._d})

            if delta < eps:
                break

    def fit(self, x: np.ndarray, lr: float = 0.01, sigma: float = 1, eps: float = 0.01):

        self._N1, self._N2 = x.shape

        self._V = np.random.normal(0, 1/lr, (self._d, self._N2))
        self._U = np.zeros((self._d, self._N1))

        self._pmf(x, lr, sigma, eps)

        self._score = mean_squared_error(x.reshape(-1), self.transform().reshape(-1)) ** 0.5

        return self

    def transform(self):
        return np.round(np.dot(self._U.T, self._V))

    def fit_transform(self, x: np.ndarray, lr: float = 0.01, sigma: float = 1, eps: float = 0.01):
        self.fit(x, lr, sigma)
        return self.transform()

    def pickle(self, add: str = ''):
        with open(os.path.join(os.getcwd(), 'U' + add + '.pkl'), 'wb') as f:
            pkl.dump(self._U, f)
        with open(os.path.join(os.getcwd(), 'V' + add + '.pkl'), 'wb') as f:
            pkl.dump(self._V, f)

    def __str__(self):
        return MSG.format(val=self._score)

    @property
    def cost(self):
        return self._cost

    @property
    def parameters(self):
        return self._parameters

    @property
    def score(self):
        return self._score


def form_matrix(table: pd.DataFrame, col1: str, col2: str, target_col: str):
    x = np.zeros((np.max(table[col1]), np.max(table[col2])))
    print(x.shape)
    for i in range(len(table)):
        x[table[col1][i]-1, table[col2][i]-1] = table[target_col][i]
    return x


if __name__ == '__main__':

    ITERATIONS = 15
    SIGMA = 24
    EPS = 10e-2

    data_artists = pd.read_table(os.path.join(os.getcwd(), 'data', 'user_artists.dat'))
    X = form_matrix(data_artists, 'userID', 'artistID', 'weight')

    """
    scores = []
    
    for d in [1, 5, 10, 20, 100]:
        model = PMF(
            d_param=d,
            max_iter_number=ITERATIONS
        )

        model.fit(X[:, 1:2000 + 1], lr=4, sigma=SIGMA, eps=EPS)
        print(model)

        scores.append(model.score)
        model.pickle(str(d))

    print(np.argmin(scores)+1, np.min(scores))
    """
