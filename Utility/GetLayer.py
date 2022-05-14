
from tensorflow.keras import layers
from tqdm import tqdm
def get_layer(name, units, dropout, return_state=False, return_sequences=False):
    """Case when cell type is simple RNN"""
    if name=="rnn":
        return layers.SimpleRNN(units=units, dropout=dropout, 
                                return_state=return_state,
                                return_sequences=return_sequences)
        
    """Case when cell type is simple GRU"""
    if name=="gru":
        return layers.GRU(units=units, dropout=dropout, 
                          return_state=return_state,
                          return_sequences=return_sequences)
        
    """Case when cell type is simple LSTM"""
    if name=="lstm":
        return layers.LSTM(units=units, dropout=dropout, 
                           return_state=return_state,
                           return_sequences=return_sequences)
