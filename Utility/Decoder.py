
import tensorflow as tf
from tensorflow.keras import layers
from Utility.Attention import BahdanauAttention
from Utility.GetLayer import get_layer
from Utility.Param import Parameters


"""This class contain all funtion to add Decoder layers
Input: Param 
This variable contain all configuration details.But most focus goes on following attrib
- layer_type
- encoder_layers
- units
- dropout
- Attention [True,False]
-  attention_type [Bahdanau]
"""
#TODO : Implement Luong attention

class Decoder(tf.keras.Model):
    def __init__(self,param):

        super(Decoder, self).__init__()

        #Basic configurations
        self.layer_type = param.layer_type
        self.n_layers = param.decoder_layers
        self.units =param. units
        self.dropout = param.dropout

        #Following configuration useful in case of attention enabled model
        # attention_type = [Luong,Bahdanau]
        self.attention = param.attention
        
        self.attention_type=param.attention_type

        #Add embedding layers 
        self.embedding_layer = layers.Embedding(input_dim=param.decoder_vocab_size, 
                                                output_dim=param.embedding_dim,trainable=True)
        
        self.dense = layers.Dense(param.decoder_vocab_size, activation="softmax")
        self.flatten = layers.Flatten()

        #Verify If want to add attention layers 
        #it will be  Bahdanau attention
    
        if self.attention:

            self.attention_layer = BahdanauAttention(self.units)

        #Add one/more recurrant layers based on confugurations
        self.create_rnn_layers()

    def call(self, x, hidden, enc_out=None):
        #Add embedding input layer
        x = self.embedding_layer(x)

        #Verify if attention layer need to be added
        if self.attention:
            context_vector, attention_weights = self.attention_layer(hidden, enc_out)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], -1)
        else:
            attention_weights = None

        
        x = self.rnn_layers[0](x, initial_state=hidden)

        #Get returned output and state value
        for layer in self.rnn_layers[1:]:
            x = layer(x)

        output, state = x[0], x[1:]

        output = self.dense(self.flatten(output))
        
        return output, state, attention_weights
    #Create decoder layesr
    def create_rnn_layers(self):
        self.rnn_layers = []   

        #Add one or more decoder layers

        for i in range(self.n_layers - 1):
            self.rnn_layers.append(get_layer(self.layer_type, self.units, self.dropout,
                                                return_sequences=True,
                                                return_state=True))
        
        self.rnn_layers.append(get_layer(self.layer_type, self.units, self.dropout,
                                            return_sequences=False,
                                            return_state=True))