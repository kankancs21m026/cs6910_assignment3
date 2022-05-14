import tensorflow as tf
from tensorflow.keras import layers
from Utility.GetLayer import get_layer
from Utility.Param import Parameters

"""This class contain all funtion to add Encoder layers
Input: Param 
This variable contain all configuration details.But most focus goes on following attributes:
- layer_type
- encoder_layers
- units
- dropout
"""
class Encoder(tf.keras.Model):
    def __init__(self, param):
        #Configurations
        super(Encoder, self).__init__()
        self.layer_type = param.layer_type
        self.n_layers = param.encoder_layers
        self.units = param.units
        self.dropout = param.dropout
        self.embedding = tf.keras.layers.Embedding(param.encoder_vocab_size, param.embedding_dim,trainable=True)

        #Create Recurrant Layers
        self.create_rnn_layers()

    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.rnn_layers[0](x, initial_state=hidden)

        #Get returned output and state value
        for layer in self.rnn_layers[1:]:
            x = layer(x)

        output, state = x[0], x[1:]

        return output, state
    "Create Encoder layer"
    def create_rnn_layers(self):
        self.rnn_layers = []
        #Add one or more encoder layers
        for i in range(self.n_layers):
            self.rnn_layers.append(get_layer(self.layer_type, self.units, self.dropout,
                                                return_sequences=True,
                                                return_state=True))


    def initialize_hidden_state(self, batch_size):

        if self.layer_type != "lstm":
            return [tf.zeros((batch_size, self.units))]
        else:
            return [tf.zeros((batch_size, self.units))]*2



    