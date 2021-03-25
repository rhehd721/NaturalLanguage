import tensorflow as tf
from tensorflow.keras import preprocessing

samples = ['너 오늘 이뻐 보인다',
            '나는 오늘 기분이 더러워',
            '끝내주는데, 좋은 일이 있나봐',
            '나 좋은 일이 생겼어',
            '아 오늘 진짜 짜증나',
            '환상적인데, 정말 좋은거 같아']

labels = [[1],[0],[1],[1],[0],[1]]
tokenizer = preprocessing.text.Tokenizer() tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index