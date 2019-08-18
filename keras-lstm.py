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

# ### 以下のように入力文をjoinする  
# [PAD] (input1) [SEP] (input2) [CLS]

data['input'] = '[PAD] ' + data['jap_question_1_wakati'] + ' [SEP] ' + data["jap_question_2_wakati"] + ' [CLS]'
X = data[['input']]
y = data[['is_duplicate']]

X.head()

# ### 最大文字長の指定

# %matplotlib inline

# 分布の確認
X['input'].apply(lambda x: len(x.split(' '))).hist()

max_sentence_length = 200

# 最大文字長以上のデータを削除
condition = X['input'].apply(lambda x: len(x.split(' '))) <= max_sentence_length
X = X[condition]
y = y[condition]

print(X.shape)
print(y.shape)

# ### 他のモデルと性能を比較するために単語辞書は別で定義しておく
# 注: BERTと比較したいので、BERTのpretrainedモデルの辞書を使用

with open('vocab.txt', 'r') as f:
    vocab = f.read()

id2vocab = {i: val for i, val in enumerate(vocab.split('\n'))}
vocab2id = {g: f for f, g in id2vocab.items()}


# +
def replace_id(text):
    return " ".join([str(vocab2id.get(f, vocab2id['[UNK]'])) for f in text.split(" ")])

X = X.assign(
    input=lambda df: df['input'].apply(replace_id)
)
# -

X_list = [[int(g) for g in f.split(' ')] for f in X['input'].to_list()]

# ### ゼロ埋め
# 全ての入力文が最大文字長になるように足りない文字数だけゼロ埋め

from keras.preprocessing.sequence import pad_sequences

X_pre = pad_sequences(X_list, maxlen=max_sentence_length)

y_pre = pd.get_dummies(y['is_duplicate']).to_numpy()

# # model
# LSTMを使用

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# +
num_words = len(id2vocab)
emb_dim = 128
num_labels = 2
n_hidden = 200

def make_model(len_seq, emb_dim, num_labels, n_hidden):
    model = Sequential()
    model.add(Embedding(num_words, emb_dim, input_length=len_seq))
    model.add(LSTM(n_hidden, batch_input_shape=(None, len_seq, emb_dim), return_sequences=False))
    model.add(Dense(num_labels))
    model.add(Activation("sigmoid"))
    return model

model = make_model(
    len_seq=max_sentence_length,
    emb_dim=emb_dim,
    num_labels=num_labels,
    n_hidden=n_hidden
)
# -

model.summary()

# # train and evaluation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_pre, y_pre, shuffle=True, random_state=123)

optimizer = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1)
