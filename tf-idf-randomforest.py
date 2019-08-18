# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# # データ取得

data = pd.read_csv('data/train.csv')

data = data[['jap_question_1_wakati', 'jap_question_2_wakati', 'is_duplicate']]

data.info()

# # tf-idfの取得
# 特徴量として使うためにtf-idfを計算  
# 2つの入力文それぞれに対してtf-idf求め、2つを連結させて特徴量とする

data_all = pd.concat([data['jap_question_1_wakati'], data['jap_question_1_wakati']], axis=0
                    ).to_numpy()

from sklearn.feature_extraction.text import TfidfVectorizer

# +
vec_tfidf = TfidfVectorizer()

vec_tfidf.fit(data_all)

# +
data1 = vec_tfidf.transform(data['jap_question_1_wakati'].to_numpy())
data2 = vec_tfidf.transform(data['jap_question_2_wakati'].to_numpy())

assert data1.shape == data2.shape and data1.shape[0] == len(data)

# +
from scipy.sparse import hstack
X = hstack([data1, data2])

assert X.shape == (data1.shape[0], data1.shape[1] * 2)

# +
y = data[['is_duplicate']].to_numpy()

assert y.dtype == np.int64
# -

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123)

X_train.shape[0] / X.shape[0]

# # model definition
# random forestで2値分類(pair or not)

model = RandomForestClassifier()

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print('train acc: ', train_score)
print('test acc:', test_score)

# # classification_report
# accuracy, precision, recall, f1-scoreなどを算出

from sklearn.metrics import classification_report

# +
# train metrics
y_pred = model.predict(X_train)
rep_train = classification_report(y_train, y_pred)

print(rep_train)

# +
# test metrics
y_pred = model.predict(X_test)
rep = classification_report(y_test, y_pred)

print(rep)
