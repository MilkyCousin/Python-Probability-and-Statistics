import os

import pandas as pd
import numpy as np
import scipy.stats as sc

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Варiант 4
# Данi: crx.data, crx.names
# Вiдгук: Остання колонка
# Факторна змiнна: A13

np.random.seed(0)


def useless_strings(msg: str, n: int = 20, sep: str = '='):
    m = (2*n + len(msg))
    print(sep * m + '\n' + sep * n + msg + sep * n + '\n' + sep * m)


def binary_classification_report(y_true, y_pred, name, p=True):
    acc = accuracy_score(y_true, y_pred)
    if p:
        useless_strings(name, sep='#', n=8)
        print(classification_report(y_true, y_pred))
        cont_matrix = confusion_matrix(y_true, y_pred)
        plt.title(name)
        sb.heatmap(cont_matrix, annot=True)
        #plt.show()
        plt.savefig(current_path + os.sep + 'pictures' + os.sep + 'test3' + os.sep + 'confusion_' + name + '.png')
        plt.clf()
    return acc


def fast_model_fit_and_prediction(model, Xtr, ytr, Xte, yte, name='R', p=True):
    model.fit(Xtr, ytr)
    ypr = model.predict(Xte)
    accuracy = binary_classification_report(yte, ypr, name, p=p)
    return accuracy


def knn_cv_by_accuracy(Xtr, ytr, Xte, yte, low_lim, up_lim, dist='euclidean', dist_params=None):
    knn_acc = []

    for n in range(low_lim, up_lim + 1):
        cls_knn = KNeighborsClassifier(
            n_neighbors=n, metric=dist,
            metric_params=dist_params
        )

        current_acc = fast_model_fit_and_prediction(
            cls_knn, Xtr, ytr, Xte, yte, name=' KNN #%i ' % n, p=False
        )

        knn_acc.append(current_acc)

    plt.plot(knn_acc)
    plt.savefig(current_path + os.sep + 'pictures' + os.sep + 'test3' + os.sep + 'knn_plot_' + dist + '.png')
    #plt.show()
    plt.clf()

    opt_n = np.argmax(knn_acc) + 1

    return opt_n, np.max(knn_acc)


def test_template(Xtr, ytr, Xte, yte):
    cls_gaussian_nb = GaussianNB()

    nb_acc = fast_model_fit_and_prediction(
        cls_gaussian_nb, Xtr, ytr, Xte, yte, name=' NB#1 '
    )

    print('Accuracy of Gaussian NB: %.4f' % nb_acc)

    knn_m = knn_cv_by_accuracy(
        Xtr, ytr, Xte, yte, low_lim=2, up_lim=int(np.ceil(len(Xtr) * 0.25)),
        dist='mahalanobis', dist_params={'V': Xtr.cov()}
    )

    print('Metric: Mahalanobis')
    print('Optimal number of neighbors: %i\nAccuracy of opt KNN: %.4f' % (knn_m[0], knn_m[1]))

    knn_e = knn_cv_by_accuracy(
        Xtr, ytr, Xte, yte, low_lim=2, up_lim=int(np.ceil(len(Xtr) * 0.25))
    )

    print('Metric: Euclidean')
    print('Optimal number of neighbors: %i\nAccuracy of opt KNN: %.4f' % (knn_e[0], knn_e[1]))

    knn_opt = fast_model_fit_and_prediction(
        KNeighborsClassifier(n_neighbors=knn_e[0] if knn_e[1] > knn_m[1] else knn_m[0], metric='euclidean'),
        Xtr, ytr, Xte, yte, name=' KNN#1 '
    )

    cls_logit = LogisticRegression(random_state=0, max_iter=10e3)

    logit_acc = fast_model_fit_and_prediction(
        cls_logit, Xtr, ytr, Xte, yte, name=' LOGIT#1 '
    )

    return nb_acc, knn_opt, logit_acc


current_path = os.getcwd()
frame_name = r'crx.data'

df_crx = pd.read_table(current_path + os.sep + frame_name, sep=',')

col_names = ['A%i' % i for i in range(1, df_crx.shape[1]+1)]
factor_col = 'A13'
feature_col = 'A16'

col_names_without_factor = col_names.copy()
col_names_without_factor.remove(factor_col)

col_names_without_feature = col_names_without_factor.copy()
col_names_without_feature.remove(feature_col)

col_names_continuous = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']

df_crx.columns = col_names

useless_strings(' DATA PREPARATION ')

print(f'Before cleanup: {df_crx.shape}')

na_char = '?'

for col_name in df_crx.columns:
    if na_char in set(df_crx[col_name]):
        df_crx = df_crx[df_crx[col_name] != na_char]

print(f'After cleanup: {df_crx.shape}')

# Мне самому страшно стало от написанного

mappings_binary = {
    '+': 1, '-': 0, 't': 1, 'f': 0, 'b': 1, 'a': 0
}

df_crx = df_crx.replace(
    {
        'A1': mappings_binary,
        'A9': mappings_binary,
        'A10': mappings_binary,
        'A12': mappings_binary,
        'A16': mappings_binary
    }
)

df_crx = df_crx.replace(
    {
        'A6': {
            'c': 0, 'd': 1, 'cc': 2, 'i': 3, 'j': 4, 'k': 5, 'm': 6,
            'r': 7, 'q': 8, 'w': 9, 'x': 10, 'e': 11, 'aa': 12, 'ff': 13
        },
        'A7': {
            'v': 0, 'h': 1, 'bb': 2, 'j': 3, 'n': 4,
            'z': 5, 'dd': 6, 'ff': 7, 'o': 8
        }
    }
)

df_crx = df_crx.replace(
    {
        'A4': {
            'u': 0, 'y': 1, 'l': 2, 't': 3
        },
        'A5': {
            'g': 0, 'p': 1, 'gg': 3
        }
    }
)

print(f'Before recasting:\n{df_crx.dtypes}')

for col_name in col_names_without_factor:
    df_crx[col_name] = df_crx[col_name].astype('float64')

print(f'After recasting:\n{df_crx.dtypes}')

df_crx_train, df_crx_test = train_test_split(df_crx, train_size=0.8, random_state=0)

alpha = 0.001
accepted = []

for col_name in col_names_continuous:
    stat_1, p_value_1 = sc.kstest(df_crx[col_name], 'norm', N=len(df_crx[col_name]))
    print(
        'KS-test for normality: [%s], stat: %4.4f, p-value: %4.4f' % (col_name, stat_1, p_value_1)
    )
    stat_2, p_value_2 = sc.shapiro(df_crx[col_name])
    print(
        'Shapiro for normality: [%s], stat: %4.4f, p-value: %4.4f' % (col_name, stat_2, p_value_2)
    )
    if (p_value_1 < alpha) and (p_value_2 < alpha):
         accepted.append(col_name)

print(accepted)

f, a = plt.subplots(1, len(accepted), figsize=(28,6))

for i, col_name in enumerate(accepted):
    a[i].title.set_text(col_name)
    a[i].hist(df_crx[col_name])

plt.show()
#plt.savefig(current_path + os.sep + 'pictures' + os.sep + 'hist_norm' + '.png')


# Только непрерывные, всё множество
"""
useless_strings(' CLASSIFICATION: ONLY CONTINUOUS DATA ')

res1 = test_template(
    df_crx_train[col_names_continuous], df_crx_train[feature_col],
    df_crx_test[col_names_continuous], df_crx_test[feature_col]
)

# Все колонки, всё множество

useless_strings(' CLASSIFICATION: FULL DATA ')

res2 = test_template(
    df_crx_train[col_names_without_feature], df_crx_train[feature_col],
    df_crx_test[col_names_without_feature], df_crx_test[feature_col]
)
"""
useless_strings(' CLASSIFICATION: ELEMENT-WISE ')

for key in set(df_crx[factor_col]):
    print(
        'Number of rows after filtering bu factor [%s]:\n'
        'Train: %i\n'
        'Test: %i' % (key, sum(df_crx_train['A13'] == key), sum(df_crx_test['A13'] == key))
    )

useless_strings(' CLASSIFICATION: COMBINATIONS ')

res3 = []

for key in [['s', 'g'], ['p', 'g']]:
    useless_strings(str(key))

    df_crx_train_filtered = df_crx_train[(df_crx_train['A13'] == key[0]) | (df_crx_train['A13'] == key[1])]
    df_crx_test_filtered = df_crx_test[(df_crx_test['A13'] == key[0]) | (df_crx_test['A13'] == key[1])]

    res3.append(
        test_template(
            df_crx_train_filtered[col_names_without_feature], df_crx_train_filtered[feature_col],
            df_crx_test_filtered[col_names_without_feature], df_crx_test_filtered[feature_col]
        )
    )
"""
df_results = pd.DataFrame(
    {'TEST1': res1, 'TEST2': res2, 'TEST3_1': res3[0], 'TEST3_2': res3[1]},
    index=pd.Index(['GaussianNB', 'KNN', 'Logistic'], name="rows")
)
print(df_results)
"""