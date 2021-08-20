#!/usr/bin/env python
# coding: utf-8

# Скачайте датасет
# https://www.kaggle.com/team-ai/spam-text-message-classification
# що мiстить набiр повiдомлень (спамових або нi) та побудуйте спам-фiльтр на основi
# наївного байєсiвського класифiкатора.
# https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering.

# In[110]:


import os

from collections import Counter, defaultdict 
from string import punctuation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


np.random.seed(0)


# In[3]:


lemmatizer = WordNetLemmatizer() 
stop_words = set(stopwords.words('english'))


# In[4]:


df = pd.read_csv('dataset_spam.csv')


# In[5]:


df.columns = ['label', 'text']


# In[6]:


df.head(5)


# Побудуємо наївний байєсівський класифікатор с використанням tf-idf матриці частот поширених слів набору.
# На тренувальну та тестову вибірку буде виділено 80% та 20% від усього набору даних відповідно.

# In[7]:


df['label'] = df.replace({
    'label': {'ham': 0, 'spam': 1}
})
df['label'] = df['label'].astype('bool')


# In[8]:


df.head(5)


# Побудуємо словник, вилучимо зайві слова та пукнтуацію

# In[9]:


def preprocess_text(tokenizer, lemmatizer, stop_words, punctuation, text): 
    tokens = tokenizer(text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return [token for token in lemmas if token not in stop_words and token not in punctuation]

df['cleaned'] = df['text'].apply(
    lambda x: preprocess_text(
        word_tokenize, lemmatizer, stop_words, punctuation, x
    )
)


# In[10]:


df.head(5)


# In[11]:


def flat_nested(nested):
    flatten = []
    for item in nested:
        if isinstance(item, list):
            flatten.extend(item)
        else:
            flatten.append(item)
    return flatten


# In[12]:


vocab = set(flat_nested(df.cleaned.tolist()))


# In[13]:


len(vocab)


# In[14]:


cnt_vocab = Counter(flat_nested(df.cleaned.tolist()))


# In[15]:


cnt_vocab.most_common(15)


# In[16]:


def bar_plot_sizes(frame, col_name, a = 6, b = 8):
    
    sizes = [
        len(frame[col_name][frame[col_name] == 0]), 
        len(frame[col_name][frame[col_name] == 1])
    ]
    
    plt.figure(figsize = (a, b))
    
    plt.title('Number of texts per type')
    plt.ylabel('Number of occurences', fontsize = 12)
    plt.xlabel('Text type', fontsize = 12)
    
    axes_current = sb.barplot(x = ['non-labeled', 'labeled'], y = sizes)
    
    plot_objects = axes_current.patches
    plot_labels = sizes
    for rect, label in zip(plot_objects, plot_labels):
        height = rect.get_height()
        axes_current.text(
            rect.get_x() + rect.get_width() / 2, 
            height + 5, label, 
            ha = 'center', va = 'bottom')

    plt.show()
    
bar_plot_sizes(df, 'label')


# На графіку видно, що в наборі кількість текстів з поміткою "спам" невелика. 
# Спробуємо отримати результати з першої моделі, а потім на вибірках, побудованих за допомогою стратифікації.

# In[17]:


from nltk.probability import FreqDist

plt.figure(figsize = (12, 6))
FreqDist(flat_nested(df['cleaned'].tolist())).plot(15, cumulative = False)


# In[18]:


from scipy.stats import describe
freqs = np.array([c[1] for c in cnt_vocab.most_common()])
print(describe(freqs))


# In[19]:


threshold_count = 5
threshold_len = 3 
cleaned_vocab = [
    token for token, count in cnt_vocab.items() if count > threshold_count and len(token) > threshold_len
]


# In[20]:


cleaned_vocab = [word for word in cleaned_vocab if not set(punctuation) & set(word)]


# In[21]:


len(cleaned_vocab)


# # Перша модель

# In[22]:


p = 0.8
df_train, df_test = train_test_split(df, train_size=p, random_state=0)


# In[23]:


text_vectorizer = TfidfVectorizer(vocabulary=cleaned_vocab)


# In[24]:


Xtr, ytr = text_vectorizer.fit_transform([' '.join(e) for e in df_train['cleaned']]), df_train['label']
Xte, yte = text_vectorizer.fit_transform([' '.join(e) for e in df_test['cleaned']]), df_test['label']


# In[25]:


cls = GaussianNB()
cls.fit(Xtr.toarray(), ytr)


# In[26]:


ypr = cls.predict(Xte.toarray())


# In[27]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte, ypr), confusion_matrix(yte, ypr, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[28]:


print(classification_report(yte, ypr))
print(accuracy_score(yte, ypr))


# # Друга модель

# Збереження пропорцій між двома вибірками. Результати незначно погіршилися. Це пояснюється зміною обсягів тренувальної та тестової вибірки.

# In[29]:


df_train_s, df_test_s = train_test_split(df, stratify=df['label'], random_state=0)


# In[30]:


Xtr_s, ytr_s = text_vectorizer.fit_transform([' '.join(e) for e in df_train_s['cleaned']]), df_train_s['label']
Xte_s, yte_s = text_vectorizer.fit_transform([' '.join(e) for e in df_test_s['cleaned']]), df_test_s['label']


# In[31]:


cls = GaussianNB()
cls.fit(Xtr_s.toarray(), ytr_s)


# In[32]:


ypr_s = cls.predict(Xte_s.toarray())


# In[33]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte_s, ypr_s), confusion_matrix(yte_s, ypr_s, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[34]:


print(classification_report(yte_s, ypr_s))
print(accuracy_score(yte_s, ypr_s))


# # Модель 3

# In[35]:


cls = MultinomialNB()
cls.fit(Xtr.toarray(), ytr)


# In[36]:


ypr = cls.predict(Xte.toarray())


# In[37]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte, ypr), confusion_matrix(yte, ypr, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[38]:


print(classification_report(yte, ypr))
print(accuracy_score(yte, ypr))


# # Модель 4

# In[39]:


cnt_vectorizer = CountVectorizer(vocabulary=cleaned_vocab)


# In[40]:


Xtr_c, ytr_c = cnt_vectorizer.fit_transform([' '.join(e) for e in df_train['cleaned']]), df_train['label']
Xte_c, yte_c = cnt_vectorizer.fit_transform([' '.join(e) for e in df_test['cleaned']]), df_test['label']


# In[41]:


cls = MultinomialNB()
cls.fit(Xtr_c.toarray(), ytr_c)


# In[42]:


ypr_c = cls.predict(Xte_c.toarray())


# In[43]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte_c, ypr_c), confusion_matrix(yte_c, ypr_c, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[44]:


print(classification_report(yte_c, ypr_c))
print(accuracy_score(yte_c, ypr_c))


# # Висновки

# Використано дві моделі наївної байєсівської класифікації на текстових даних. MultinomialNB з використанням tf-idf матриці (або навіть матриці частот) показує накращі показники.

# # Продовження

# # Опорна машина векторів

# In[45]:


svm_cls = SVC(C=1, kernel='rbf', gamma='scale', random_state=0)
svm_cls.fit(Xtr.toarray(), ytr)


# In[46]:


ypr = svm_cls.predict(Xte.toarray())


# In[47]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte, ypr), confusion_matrix(yte, ypr, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[48]:


print(classification_report(yte, ypr))
print(accuracy_score(yte, ypr))


# # AdaBoost + Tree

# In[102]:


dec_cls = DecisionTreeClassifier(criterion='entropy', random_state=0)


# In[103]:


path = dec_cls.cost_complexity_pruning_path(Xtr, ytr)

alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(alphas[:-1], impurities[:-1], color='orange', drawstyle="steps-post")
ax.set_xlabel("Alpha values")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set plot")

tree_classifiers = []

for alpha in alphas:
    dec_cls_i = DecisionTreeClassifier(random_state=0, criterion='entropy', ccp_alpha=alpha)
    dec_cls_i.fit(Xtr, ytr)
    tree_classifiers.append(dec_cls_i)

tree_classifiers = tree_classifiers[:-1]
alphas = alphas[:-1]

test_scores = [
    clf.score(Xte, yte) for clf in tree_classifiers
]

opt_alpha = alphas[np.argmax(test_scores)]


# In[104]:


dec_cls = DecisionTreeClassifier(criterion='entropy', ccp_alpha=opt_alpha, random_state=0)
dec_cls.fit(Xtr.toarray(), ytr)


# In[105]:


ypr = dec_cls.predict(Xte.toarray())


# In[106]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte, ypr), confusion_matrix(yte, ypr, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[107]:


print(classification_report(yte, ypr))
print(accuracy_score(yte, ypr))


# # Random Forest

# In[111]:


rnf_cls = RandomForestClassifier(criterion='entropy', random_state=0)
rnf_cls.fit(Xtr.toarray(), ytr)


# In[112]:


ypr = rnf_cls.predict(Xte.toarray())


# In[113]:


f, a = plt.subplots(1,2, figsize=(20,8))
m1, m2 = confusion_matrix(yte, ypr), confusion_matrix(yte, ypr, normalize='true')
a[0].title.set_text('absolute frequency')
sb.heatmap(m1, annot=True, ax=a[0])
a[1].title.set_text('relative frequency')
sb.heatmap(m2, annot=True, ax=a[1])
plt.show()
print(m1)
print(m2)


# In[114]:


print(classification_report(yte, ypr))
print(accuracy_score(yte, ypr))


# # Висновки

# Випадковий ліс та опорна машина векторів якісно впоралися з задачею класифікації. 
