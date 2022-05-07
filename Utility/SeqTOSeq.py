import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import random 
import time 
import wandb
import numpy as np
import pandas as pd 
class SequenceTOSequence():
    def __init__(self, parameters):
        self.embedding_dim = parameters.embedding_dim
        self.encoder_layers = parameters.encoder_layers
        self.decoder_layers = parameters.decoder_layers
        self.layer_type = parameters.layer_type
        self.units = parameters.units
        self.dropout = parameters.dropout
        self.attention = parameters.attention
        self.stats = []
        self.batch_size = parameters.batch_size
        self.apply_beam_search = parameters.apply_beam_search
        self.restoreBestModel=parameters.restoreBestModel
        self.patience=parameters.patience
    def build(self, loss, metric,optimizer='adam',lr=0.001):
        self.loss = loss
        if(optimizer=='adam'):
          self.optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        if(optimizer=='nadam'):
          self.optimizer=tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
          self.optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr)
         
        self.metric = metric

    def set_vocabulary(self, input_tokenizer, targ_tokenizer):
        self.input_tokenizer = input_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.create_model()
    
  
    def create_model(self):

        encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        decoder_vocab_size = len(self.targ_tokenizer.word_index) + 1

        self.encoder = Encoder(self.layer_type, self.encoder_layers, self.units, encoder_vocab_size,
                               self.embedding_dim, self.dropout)

        self.decoder = Decoder(self.layer_type, self.decoder_layers, self.units, decoder_vocab_size,
                               self.embedding_dim,  self.dropout, self.attention)

    @tf.function
    def train(self, input, target, enc_state):

        loss = 0 

        with tf.GradientTape() as tape: 

            enc_out, enc_state = self.encoder(input, enc_state)

            dec_state = enc_state
            dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

            ## We use Teacher forcing to train the network
            ## Each target at timestep t is passed as input for timestep t + 1

            if  (self.apply_teacher_forcing==True):
              
                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)
                    
                    dec_input = tf.expand_dims(target[:,t], 1)
            
            else:

                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)

                    preds = tf.argmax(preds, 1)
                    dec_input = tf.expand_dims(preds, 1)


            batch_loss = loss / target.shape[1]

            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, self.metric.result()

    @tf.function
    def validation(self, input, target, enc_state):

        loss = 0
        
        enc_out, enc_state = self.encoder(input, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

        for t in range(1, target.shape[1]):

            preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
            loss += self.loss(target[:,t], preds)
            self.metric.update_state(target[:,t], preds)

            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        batch_loss = loss / target.shape[1]
        
        return batch_loss, self.metric.result()
  

    def fit(self, dataset, val_dataset, batch_size=128, epochs=5, wandb=None, apply_teacher_forcing=True):

        self.batch_size = batch_size
        self.apply_teacher_forcing = apply_teacher_forcing

        steps_per_epoch = len(dataset) // self.batch_size
        steps_per_epoch_val = len(val_dataset) // self.batch_size
        
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=False)

      
        sample_inp, sample_targ = next(iter(dataset))
        self.max_target_len = sample_targ.shape[1]
        self.max_input_len = sample_inp.shape[1]

        
        self.bestEncoder=self.encoder
        self.bestDecoder=self.decoder
        self.bestoptimizer=self.optimizer
        
        accuracyDegradePatience=0
        self.oldaccuracy=0
        for epoch in  tqdm(range(1, epochs+1), total = epochs,desc="Epochs "):
             
            if(accuracyDegradePatience>=self.patience):
                self.encoder=self.bestEncoder
                self.decoder=self.bestDecoder
                self.optimizer=self.bestoptimizer
                break
            ## Training loop ##
            total_loss = 0
            total_acc = 0
            self.metric.reset_states()

            starting_time = time.time()
            enc_state = self.encoder.initialize_hidden_state(self.batch_size)

            
           
            for batch, (input, target) in enumerate(dataset.take(steps_per_epoch)):
               
                batch_loss, acc = self.train(input, target, enc_state)
                total_loss += batch_loss
                total_acc += acc
            batch_loss, acc = self.train(input, target, enc_state)
            avg_acc = total_acc / steps_per_epoch
            avg_loss = total_loss / steps_per_epoch

            # Validation loop ##
            total_val_loss = 0
            total_val_acc = 0
            self.metric.reset_states()

            enc_state = self.encoder.initialize_hidden_state(self.batch_size)

            
            for batch, (input, target) in enumerate(val_dataset.take(steps_per_epoch_val)):
                batch_loss, acc = self.validation(input, target, enc_state)
                total_val_loss += batch_loss
                total_val_acc += acc

            avg_val_acc = total_val_acc / steps_per_epoch_val
            avg_val_loss = total_val_loss / steps_per_epoch_val
            if(self.oldaccuracy>avg_val_acc):
              accuracyDegradePatience+=1
            else:
              self.bestEncoder=self.encoder
              self.bestDecoder=self.decoder
              self.bestoptimizer=self.optimizer
              self.oldaccuracy=avg_val_acc
            print( "\nTrain Loss: {0:.4f} Train Accuracy: {1:.4f} Validation Loss: {2:.4f} Validation Accuracy: {3:.4f}".format(avg_loss, avg_acc*100, avg_val_loss, avg_val_acc*100))
            
            time_taken = time.time() - starting_time
            self.stats.append({"epoch": epoch,
                            "train_loss": avg_loss,
                            "val_loss": avg_val_loss,
                            "train_acc": avg_acc*100,
                            "val_acc": avg_val_acc*100,
                            "training time": time_taken})
            
            if not (wandb is None):
                wandb.log(self.stats[-1])
            
            print(f"\nTime taken for the epoch {time_taken:.4f}")
           
        
        print("\nModel trained successfully !!")
        
    def evaluate(self, test_dataset, batch_size=None):

        if batch_size is not None:
            self.batch_size = batch_size

        steps_per_epoch_test = len(test_dataset) // batch_size
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
        
        total_test_loss = 0
        total_test_acc = 0
        self.metric.reset_states()

        enc_state = self.encoder.initialize_hidden_state(self.batch_size)

        print("\nRunning test dataset through the model...\n")
        for batch, (input, target) in enumerate(test_dataset.take(steps_per_epoch_test)):
            batch_loss, acc = self.validation(input, target, enc_state)
            total_test_loss += batch_loss
            total_test_acc += acc

        avg_test_acc = total_test_acc / steps_per_epoch_test
        avg_test_loss = total_test_loss / steps_per_epoch_test
    
        print(f"Test Loss: {avg_test_loss:.4f} Test Accuracy: {avg_test_acc:.4f}")

        return avg_test_loss, avg_test_acc


    def translate(self, word, get_heatmap=False):

        word = "\t" + word + "\n"

        inputs = self.input_tokenizer.texts_to_sequences([word])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_input_len,
                                                               padding="post")

        result = ""
        att_wts = []

        enc_state = self.encoder.initialize_hidden_state(1)
        enc_out, enc_state = self.encoder(inputs, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*1, 1)

        for t in range(1, self.max_target_len):

            preds, dec_state, attention_weights = self.decoder(dec_input, dec_state, enc_out)
            
            if get_heatmap:
                att_wts.append(attention_weights)
            
            preds = tf.argmax(preds, 1)
            next_char = self.targ_tokenizer.index_word[preds.numpy().item()]
            result += next_char

            dec_input = tf.expand_dims(preds, 1)

            if next_char == "\n":
                return result[:-1], att_wts[:-1]

        return result[:-1], att_wts[:-1]

    