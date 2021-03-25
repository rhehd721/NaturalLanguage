import tensorflow as tf
from tensorflow.keras import preprocessing

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, emb_size, input_length = 4))
model.add(layers.Lambda(lambda x : tf.reduce_mean(x, axis = 1)))
model.add(layers.Dense(hidden_dimension, activation = 'relu'))
model.add(layers.Dense(output_dimension, activation = 'sigmoid'))

