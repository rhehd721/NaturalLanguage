import tensorflow as tf
from tensorflow.keras import preprocessing

import numpy as np

# 학습에 사용될 재료
samples = ['너 오늘 이뻐 보인다',
          '나는 오늘 기분이 더러워',
          '끝내주는데, 좋은 일이 있나봐',
          '나 좋은 일이 생겼어',
          '아 오늘 진짜 짜증나',
          '환상적인데, 정말 좋은거 같아']

# samples의 긍정(1) 부정(0)
targets =[[1], [0], [1], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()  # Tokenizer불러오기
tokenizer.fit_on_texts(samples)  # Tokenizer를 통해 문자 데이터를 입력받아서 리스트의 형태로 변환

sequences = tokenizer.texts_to_sequences(samples) # 텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환
# {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}
# [[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]

input_sequences = np.array(sequences)   # 숫자를 numpy 형태로 변환
labels = np.array(targets)

word_index = tokenizer.word_index

from tensorflow.keras import backend
from tensorflow.keras import layers

batch_size = 2
num_epochs = 100

vocab_size = len(word_index) + 1
emb_size = 128
hidden_dimension = 256
output_dimension = 1