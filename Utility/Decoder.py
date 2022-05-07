
import tensorflow as tf
from tensorflow.keras import layers
from Utility.Attention import BahdanauAttention
from Utility.GetLayer import get_layer
class Decoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, decoder_vocab_size, embedding_dim, dropout, attention=False):
        super(Decoder, self).__init__()

        self.layer_type = layer_type
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.attention = attention
        self.embedding_layer = layers.Embedding(input_dim=decoder_vocab_size, 
                                                output_dim=embedding_dim,trainable=True)
        
        self.dense = layers.Dense(decoder_vocab_size, activation="softmax")
        self.flatten = layers.Flatten()
        if self.attention:
            self.attention_layer = BahdanauAttention(self.units)
        self.create_rnn_layers()

    def call(self, x, hidden, enc_out=None):
        
        x = self.embedding_layer(x)

        if self.attention:
            context_vector, attention_weights = self.attention_layer(hidden, enc_out)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], -1)
        else:
            attention_weights = None

        x = self.rnn_layers[0](x, initial_state=hidden)

        for layer in self.rnn_layers[1:]:
            x = layer(x)

        output, state = x[0], x[1:]

        output = self.dense(self.flatten(output))
        
        return output, state, attention_weights

    def create_rnn_layers(self):
        self.rnn_layers = []    

        for i in range(self.n_layers - 1):
            self.rnn_layers.append(get_layer(self.layer_type, self.units, self.dropout,
                                                return_sequences=True,
                                                return_state=True))
        
        self.rnn_layers.append(get_layer(self.layer_type, self.units, self.dropout,
                                            return_sequences=False,
                                            return_state=True))