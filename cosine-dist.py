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

from sklearn.model_selection import train_test_split
_, data = train_test_split(data, shuffle=True, random_state=123)

# # tf-idfの取得

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
# -

# # calculate cosine similarity
# tf-idfのcosine similarityが大きさでquestion pairsか否か判定する

from sklearn.metrics.pairwise import cosine_similarity

cos_dist = cosine_similarity(data1, data2)

cos_dist.shape

cos_dist_diag = np.diag(cos_dist)
# cos_dist_diag = np.reshape(cos_dist_diag, (cos_dist_diag.shape[0], 1))

cos_dist_diag.shape

cos = pd.Series(cos_dist_diag)

data['cos_sim']= cos

# # evaluate
# 色々な範囲でしきい値を設けて正解率を算出

# +
data = data.assign(
    over80=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.8 else 0),
    over75=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.75 else 0),
    over70=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.7 else 0),
    over65=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.65 else 0),
    over60=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.6 else 0),
    over55=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.55 else 0),
    over50=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.5 else 0),
    over45=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.45 else 0),
    over40=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.4 else 0),
    over35=lambda df: df['cos_sim'].apply(lambda x: 1 if x >= 0.35 else 0),
)

data['rand'] = pd.Series(np.random.randint(0, 2, len(data)), index=data.index )

# +
num = len(data)


acc = len(data[data['is_duplicate'] == data['rand']]) / num
print('random acc: ', acc)

for i in range(80, 30, -5):
    acc = len(data[data['is_duplicate'] == data['over{}'.format(i)]]) / num
    print('over{} acc: '.format(i), acc)
