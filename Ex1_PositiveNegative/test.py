# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.keras import preprocessing

samples = ['너 오늘 이뻐 보인다',
            '나는 오늘 기분이 더러워',
            '끝내주는데, 좋은 일이 있나봐',
            '나 좋은 일이 생겼어',
            '아 오늘 진짜 짜증나',
            '환상적인데, 정말 좋은거 같아']

labels = [[1],[0],[1],[1],[0],[1]]
tokenizer = preprocessing.text.Tokenizer() # tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

batch_size = 2
num_epochs = 100
vocab_size = len(word_index) + 1
emb_size = 128
hidden_dimension = 256
output_dimension = 1

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, emb_size, input_length = 4))
model.add(layers.Lambda(lambda x : tf.reduce_mean(x, axis = 1)))
model.add(layers.Dense(hidden_dimension, activation = 'relu'))
model.add(layers.Dense(output_dimension, activation = 'sigmoid'))

model.compile(optimizer = tf.keras.optimizer.Adam(0.001),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

model.fit(input_sequences, labels, epochs = num_epochs, batch_size = batch_size)