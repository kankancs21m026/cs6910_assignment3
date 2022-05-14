import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from Utility.Decoder import Decoder
from Utility.Encoder import Encoder
from tqdm import tqdm
import random 
import time 
import wandb
import numpy as np
import pandas as pd
class SequenceTOSequence():
    def __init__(self, parameters):

        #Basic configurations
        self.param=parameters
        self.embedding_dim = parameters.embedding_dim
        self.encoder_layers = parameters.encoder_layers
        self.decoder_layers = parameters.decoder_layers
        self.layer_type = parameters.layer_type
        self.units = parameters.units
        self.dropout = parameters.dropout
        self.batch_size = parameters.batch_size

        #Add information regarding attention layer
        self.attention = parameters.attention
        self.attention_type = parameters.attention_type

        self.stats = []
      
        self.apply_beam_search = parameters.apply_beam_search
        
        #Early stop conditions
        self.patience=parameters.patience
        self.restoreBestModel=parameters.restoreBestModel

        #teacher forcing
        self.apply_teacher_forcing=parameters.apply_teacher_forcing    
        self.teacher_forcing_ratio=parameters.teacher_forcing_ratio

    #Build model Add specific optimizers    
    def build(self, loss, metric,optimizer='adam',lr=0.001):
        self.loss = loss

        #Select specific optimizer
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
    
    """This procedure used to define Encoder Decoder Layer"""
    def create_model(self):

        encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        decoder_vocab_size = len(self.targ_tokenizer.word_index) + 1
        self.param.encoder_vocab_size=encoder_vocab_size
        self.param.decoder_vocab_size=decoder_vocab_size
        #Add Encoder layer

        self.encoder = Encoder(self.param)

        #Create decode with or without any attention layer
        #Check following properties to add attention
        # self.attention
        # self.attention_type
        self.decoder = Decoder(self.param)
                

    @tf.function
    def train(self, input, target, enc_state):

        loss = 0 

        with tf.GradientTape() as tape: 

            enc_out, enc_state = self.encoder(input, enc_state)

            dec_state = enc_state
            dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

            apply_teacher_forcing=False
            #decide whether to use teacher forcing
            if random.random() < self.teacher_forcing_ratio:
              apply_teacher_forcing=True
            ## We use Teacher forcing to train the network
            ## Each target at timestep t is passed as input for timestep t + 1
            if  (apply_teacher_forcing==True):
                #Apply teacher forcing
                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)
                    #As teacher forcing  applied we pass next target as decoder input
                    dec_input = tf.expand_dims(target[:,t], 1)
            
            else:
                #Without teacher forcing

                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)
                    #As teacher forcing not applied we pass decoder input as whatever we predict
                    preds = tf.argmax(preds, 1)
                    dec_input = tf.expand_dims(preds, 1)


            batch_loss = loss / target.shape[1]

            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, self.metric.result()

    
  

    def fit(self, dataset, val_dataset, batch_size=128, epochs=5, wandb=None,apply_teacher_forcing=True, teacher_forcing_ratio=0.7):

        self.batch_size = batch_size
        self.apply_teacher_forcing = apply_teacher_forcing
        self.teacher_forcing_ratio=teacher_forcing_ratio
        #Prepare chunk of data based on batch size provided
        steps_per_epoch = len(dataset) // self.batch_size
        #steps_per_epoch_val = len(val_dataset) // self.batch_size
        
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        #val_dataset = val_dataset.batch(self.batch_size, drop_remainder=False)

      
        sample_inp, sample_targ = next(iter(dataset))
        self.max_target_len = sample_targ.shape[1]
        self.max_input_len = sample_inp.shape[1]

        #Store Encoder ,decoder details in case model get good accuracy
        #Will be useful to restore best model
        self.bestEncoder=self.encoder
        self.bestDecoder=self.decoder
        self.bestoptimizer=self.optimizer
        
        accuracyDegradePatience=0
        self.oldaccuracy=0
        for epoch in  tqdm(range(1, epochs+1), total = epochs,desc="Epochs "):
             
            if(accuracyDegradePatience>=self.patience):
                if(self.restoreBestModel==True):
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
                #Accumulate loss and accurecy for each batch
                batch_loss, acc = self.train(input, target, enc_state)
                total_loss += batch_loss
                total_acc += acc
            #Calculate validation accurecy for current Epoch
          
            avg_acc = total_acc / steps_per_epoch
            avg_loss = total_loss / steps_per_epoch

            # Validation loop ##
            total_val_loss = 0
            total_val_acc = 0
            self.metric.reset_states()

            enc_state = self.encoder.initialize_hidden_state(self.batch_size)

            #Process data in batches
            
            avg_val_loss, avg_val_acc = self.evaluate(val_dataset,batch_size=self.batch_size)
              

           
            #Verify if model performance degrading.add()
            #In case train accurecy improved but no significant imprrovement in validation
            #Add condition for early stopping
            #Restore best model based on the input
            if(self.oldaccuracy>avg_val_acc):
              accuracyDegradePatience+=1
            else:
              self.bestEncoder=self.encoder
              self.bestDecoder=self.decoder
              self.bestoptimizer=self.optimizer
              self.oldaccuracy=avg_val_acc
              accuracyDegradePatience=0
            print( "\nTrain Loss: {0:.4f} Train Accuracy: {1:.4f} Validation Loss: {2:.4f} Validation Accuracy: {3:.4f}".format(avg_loss, avg_acc*100, avg_val_loss, avg_val_acc*100))
            
            time_taken = time.time() - starting_time

            #Add logs for WanDb
            self.stats.append({"epoch": epoch,
                            "train_loss": avg_loss,
                            "val_loss": avg_val_loss,
                            "train_acc": avg_acc*100,
                            "val_acc": avg_val_acc*100,
                            "training time": time_taken})
            
            #Log to wanDB
            if not (wandb is None):
                wandb.log(self.stats[-1])
            
            print(f"\nTime taken for the epoch {time_taken:.4f}")
           
        
        print("\nModel trained successfully !!")
    @tf.function
    def validation(self, inp, trgt, encoder_state):
        #Custom validation

        loss = 0
        #encoder input
        encoder_output, encoder_state = self.encoder(inp, encoder_state)

        #Set initial state of decoder from encoder state
        decoder_state = encoder_state
        decoder_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

        for t in range(1, trgt.shape[1]):
            #Get decoder prediction
            prediction, decoder_state, _ = self.decoder(decoder_input, decoder_state, encoder_output)
            loss += self.loss(trgt[:,t], prediction)
            self.metric.update_state(trgt[:,t], prediction)

            prediction = tf.argmax(prediction, 1)
            decoder_input = tf.expand_dims(prediction, 1)

        batch_loss = loss / trgt.shape[1]
        
        return batch_loss, self.metric.result()    
    def evaluate(self, test_dataset, batch_size=None):
        """Evaluate our model on test data"""
        if batch_size is not None:
            self.batch_size = batch_size

        #prepare chuck of data based on the batch size
        steps_per_epoch_test = len(test_dataset) // batch_size
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
        
        total_test_loss = 0
        total_test_acc = 0
        self.metric.reset_states()

        enc_state = self.encoder.initialize_hidden_state(self.batch_size)

        #print("\nRunning test dataset through the model...\n")
        #Run in batches based on the input batch size
        for batch, (input, target) in enumerate(test_dataset.take(steps_per_epoch_test)):
            batch_loss, acc = self.validation(input, target, enc_state)
            total_test_loss += batch_loss
            total_test_acc += acc

        #Caculate avarage  test accuracy and loss
        avg_test_acc = total_test_acc / steps_per_epoch_test
        avg_test_loss = total_test_loss / steps_per_epoch_test

        #Display details
        #print(f"Test Loss: {avg_test_loss:.4f} Test Accuracy: {avg_test_acc:.4f}")

        return avg_test_loss, avg_test_acc

    """ This function used to translate english world to respective language"""

    def translate(self, word, get_heatmap=False):
        #start and end token for input word
        start="\t"
        end="\n"
        word =start  + word + end

        #Tokenize input and perform  preprocessing  
        inputs = self.input_tokenizer.texts_to_sequences([word])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_input_len,
                                                               padding="post")

        result = ""
        att_wts = []

        #Process input through encoder
        enc_state = self.encoder.initialize_hidden_state(1)
        enc_out, enc_state = self.encoder(inputs, enc_state)

        # Set initial decoder sate to encoder state
        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index[start]]*1, 1)

        #Run the loop for maximum word size the target language can have
        #We get this data during training 
        for t in range(1, self.max_target_len):

            preds, dec_state, attention_weights = self.decoder(dec_input, dec_state, enc_out)
            
            #Add attention weights which is useful for generating attention heatmaps
            if get_heatmap:
                att_wts.append(attention_weights)
            
            #Pass the current prediction as input to next iteration
            preds = tf.argmax(preds, 1)

            #Accumulate target words
            next_char = self.targ_tokenizer.index_word[preds.numpy().item()]
            result += next_char

            #Decoder input for next iteration
            dec_input = tf.expand_dims(preds, 1)

            #If we receive end token stop the loop
            if next_char == end:
                break

        return result[:-1], att_wts[:-1]

    