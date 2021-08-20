import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from graphviz import Source

from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

AdaLovelaceClassifier = AdaBoostClassifier

TXT_CODE = '.txt'
PIC_CODE = '.png'

CURRENT_PATH = os.getcwd()
PATH_TEXTS = 'texts'
PATH_PICS = 'pictures'

# Варiант 4
# Данi: crx.data, crx.names
# Вiдгук: Остання колонка
# Факторна змiнна: A13

np.random.seed(0)


def estimate_results(y_predicted, y_observed, name):
    cont_matrix = confusion_matrix(y_observed, y_predicted)
    plt.title(name)
    sb.heatmap(cont_matrix, annot=True)
    plt.savefig(os.path.join(CURRENT_PATH, PATH_PICS, name + PIC_CODE))
    plt.clf()
    acc = accuracy_score(y_observed, y_predicted)
    with open(os.path.join(CURRENT_PATH, PATH_TEXTS, name + TXT_CODE), 'w') as f:
        f.write(classification_report(y_observed, y_predicted))
        f.write('\n' + str(acc) + '\n')
    return acc


def estimate_model(clf, x_train, y_train, x_test, y_test, filename='res'):
    clf.fit(x_train, y_train)
    y_train_pred, y_test_pred = clf.predict(x_train), clf.predict(x_test)
    acc1 = estimate_results(y_train_pred, y_train, filename + '-' + 'train')
    acc2 = estimate_results(y_test_pred, y_test, filename + '-' + 'test')
    return clf, acc1, acc2


frame_name = r'crx.data'

df_crx = pd.read_table(CURRENT_PATH + os.sep + frame_name, sep=',')

col_names = ['A%i' % i for i in range(1, df_crx.shape[1]+1)]
factor_col = 'A13'
feature_col = 'A16'

col_names_without_factor = col_names.copy()
col_names_without_factor.remove(factor_col)

col_names_without_feature = col_names_without_factor.copy()
col_names_without_feature.remove(feature_col)

col_names_continuous = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']

df_crx.columns = col_names

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

for col_name in col_names_without_factor:
    df_crx[col_name] = df_crx[col_name].astype('float64')

df_crx = df_crx.reindex()
train_idx, test_idx = train_test_split(df_crx.index, train_size=0.8, random_state=0)

df_crx_normalized = pd.DataFrame(
    index=df_crx.index,
    columns=col_names_without_feature,
    data=normalize(df_crx[col_names_without_feature])
)

df_crx_normalized[factor_col] = df_crx[factor_col]
df_crx_normalized[feature_col] = df_crx[feature_col]

df_train_n, df_test_n = df_crx_normalized.loc[train_idx], df_crx_normalized.loc[test_idx]
df_train, df_test = df_crx.loc[train_idx], df_crx.loc[test_idx]

# Vectors

svm_1, acc1_svm_1, acc2_svm_1 = estimate_model(
    SVC(C=12, kernel='poly', degree=5, gamma='scale', random_state=0),
    df_train_n[col_names_without_feature], df_train_n[feature_col],
    df_test_n[col_names_without_feature], df_test_n[feature_col],
    filename='svm_normalized_1'
)

# Trees

tre_1, acc1_tre_1, acc2_tre_1 = estimate_model(
    DecisionTreeClassifier(criterion='entropy', random_state=0),
    df_train_n[col_names_without_feature], df_train_n[feature_col],
    df_test_n[col_names_without_feature], df_test_n[feature_col],
    filename='tree_normalized_1'
)

path = tre_1.cost_complexity_pruning_path(
    df_train_n[col_names_without_feature], df_train_n[feature_col]
)
alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(alphas[:-1], impurities[:-1], color='purple', drawstyle="steps-post")
ax.set_xlabel("Alpha values")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set plot")
plt.savefig('tree_stats_1.png')
plt.clf()

tree_classifiers = []

for alpha in alphas:
    tre_i = DecisionTreeClassifier(random_state=0, criterion='entropy', ccp_alpha=alpha)
    tre_i.fit(df_train_n[col_names_without_feature], df_train_n[feature_col])
    tree_classifiers.append(tre_i)

tree_classifiers = tree_classifiers[:-1]
alphas = alphas[:-1]

node_counts = [clf.tree_.node_count for clf in tree_classifiers]
depth = [clf.tree_.max_depth for clf in tree_classifiers]
fig, ax = plt.subplots(2, 1)
ax[0].plot(alphas, node_counts, color='purple', drawstyle="steps-post")
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("# of nodes")
ax[0].set_title("Number of nodes for each alpha plot")
ax[1].plot(alphas, depth, color='purple', drawstyle="steps-post")
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Depth of tree")
ax[1].set_title("Depth of tree for each alpha plot")
fig.tight_layout()
plt.savefig('tree_stats_2.png')
plt.clf()

train_scores = [
    clf.score(df_train_n[col_names_without_feature],
              df_train_n[feature_col]) for clf in tree_classifiers
]

test_scores = [
    clf.score(df_test_n[col_names_without_feature],
              df_test_n[feature_col]) for clf in tree_classifiers
]

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy for each alpha")
ax.plot(alphas, train_scores, color='purple', label="Train",
        drawstyle="steps-post")
ax.plot(alphas, test_scores, color='orange', label="Test",
        drawstyle="steps-post")
ax.legend()
plt.savefig('tree_stats_3.png')
plt.clf()

opt_alpha = alphas[np.argmax(test_scores)]

tre_2, acc1_tre_2, acc2_tre_2 = estimate_model(
    DecisionTreeClassifier(
        criterion='entropy', ccp_alpha=opt_alpha, random_state=0
    ),
    df_train_n[col_names_without_feature], df_train_n[feature_col],
    df_test_n[col_names_without_feature], df_test_n[feature_col],
    filename='tree_normalized_2'
)

# Forest

frs_1, acc1_frs_1, acc2_frs_1 = estimate_model(
    RandomForestClassifier(criterion='gini', random_state=0),
    df_train_n[col_names_without_feature], df_train_n[feature_col],
    df_test_n[col_names_without_feature], df_test_n[feature_col],
    filename='woods_normalized_1'
)

frs_2, acc1_frs_2, acc2_frs_2 = estimate_model(
    RandomForestClassifier(criterion='gini', random_state=0),
    df_train[col_names_without_feature], df_train[feature_col],
    df_test[col_names_without_feature], df_test[feature_col],
    filename='woods_unprocessed_1'
)

# Ada Lovelace (??)

errors_train = []
errors_test = []

lr = 0.25
n_opt = 300

ada_1, acc1_ada_1, acc2_ada_1 = estimate_model(
    AdaLovelaceClassifier(
        DecisionTreeClassifier(
            criterion='entropy', ccp_alpha=opt_alpha, random_state=0
        ),
        learning_rate=lr, n_estimators=n_opt, random_state=0),
    df_train_n[col_names_without_feature], df_train_n[feature_col],
    df_test_n[col_names_without_feature], df_test_n[feature_col],
    filename='ada_normalized_1'
)
