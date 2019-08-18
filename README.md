# Question Pairas Discriminator
"Quora Question Pairs" Kaggle competitionのデータをgoogle spreadsheetで日本語訳したものに対して、重複質問か否かを判別する

# elaboration
## data preprocessing
- Quora Question pairsを日本語訳したものの中から学習データ(~ 24000pairs)を使用。その内2.5割を検証データとして使用。
- mecab-ipadic-neologdで分かち書き

## feature extraction & model
### 1. tf-idf cosine similarity(cosine-dist.py)
questionに対してtf-idfを算出し、入力した2つの文章間のcosine similarityを求める。
ハンドメイドでしきい値を設けて重複質問か否か判定する。

### 2. randomforest with randomforest(tf-idf-randomforest.py)
questionに対してtf-idfを算出し、入力した2つの文章のそれぞれのtf-idfを結合し、randomforestに入力する

### 3. LSTM(keras-lstm.py)
入力分を結合し、LSTMに入力する。
kerasのLSTMを使用。
BERTと比較するためにBERTのpretrainedモデルで使用されたvocabularyを使用する。
