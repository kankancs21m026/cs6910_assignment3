import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


"""
This package have Bahdanau Attention

References -
https://medium.com/deep-learning-with-keras/seq2seq-part-c-basic-encoder-decoder-a7f536f5f510
https://arxiv.org/abs/1409.0473
https://arxiv.org/abs/1911.03853
"""
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, enc_state, enc_out):
    
    enc_state = tf.concat(enc_state, 1)
    enc_state = tf.expand_dims(enc_state, 1)

    score = self.V(tf.nn.tanh(self.W1(enc_state) + self.W2(enc_out)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * enc_out
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

