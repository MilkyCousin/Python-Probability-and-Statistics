from math import erf
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


def cdf_n_std(t):
    return (1 + erf(t/(2. ** 0.5)))/2.


def cdf_normal(t, m=0, sd=1):
    return cdf_n_std((t - m)/sd)


def quartiles(x):
    # returns lower and upper quartiles
    s = np.sort(x)
    q2 = np.median(x)
    return np.median(s[s <= q2]), np.median(s[s >= q2])


def out(x):
    # returns mask for those values in sample, that are not outliers
    q1, q3 = quartiles(x)
    iqr = q3 - q1
    return (q1 - 1.5 * iqr <= x) & (x <= q3 + 1.5 * iqr)


mu, sigma = 0, 1
N = 2000

if __name__ == '__main__':
    X = nr.normal(mu, sigma, N)
    Z = nr.standard_cauchy(N)

    S = pd.DataFrame({
        'X': X, 'Z': Z
    })

    M1, M2 = abs(S) > 1, abs(S) > 3

    C1, C2 = np.sum(M1), np.sum(M2)
    print(C1, C2, sep='\n')

    X_F = X[out(X)]
    Z_F = Z[out(Z)]

    N_BINS = 42

    plt.hist(X_F, bins=N_BINS)
    plt.title('N(%.2f, %.2f) - filtered' % (mu, sigma))
    plt.show()

    plt.hist(Z_F, bins=N_BINS)
    plt.title('Cauchy - filtered')
    plt.show()

    E_CDF_X, E_CDF_Z = ECDF(X), ECDF(Z)

    LS_X, LS_Z = np.linspace(np.min(X_F), np.max(X_F)), np.linspace(np.min(Z_F), np.max(Z_F))

    E_X, E_Z = E_CDF_X(LS_X), E_CDF_Z(LS_Z)

    plt.step(LS_X, E_X)
    plt.title('N(%.2f, %.2f) - Empirical CDF' % (mu, sigma))
    plt.show()

    plt.step(LS_Z, E_Z)
    plt.title('Cauchy - Empirical CDF')
    plt.show()

    f, ax = plt.subplots(1, 1)
    f.suptitle('Cauchy and Normal Empirical CDF', y=1)
    ax.step(LS_X, E_X, color='purple', label='Normal Empirical CDF', where='post')
    ax.step(LS_Z, E_Z, color='orange', label='Cauchy Empirical CDF', where='post')
    ax.legend()
    plt.show()

    S_MEAN, S_SD = S.mean(), S.std()

    print(S_MEAN, S_SD, sep='\n')

    P1_E = 1 - E_CDF_X(S_MEAN.X + S_SD.X * 3) + E_CDF_X(S_MEAN.X - S_SD.X * 3)
    P1_T = 1 - cdf_normal(S_MEAN.X + S_SD.X * 3) + cdf_normal(S_MEAN.X - S_SD.X * 3)

    print('E Normal CDF: %.4f\n'
          'T Normal CDF: %.4f' % (P1_E, P1_T))
