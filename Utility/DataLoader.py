import tensorflow as tf
import random 
import numpy as np
import os

from os.path import exists
import xtarfile as tarfile
import pandas as pd 
import keras
START_TOKEN="\t"
END_TOKEN="\n"


"""Download dataset if not exists"""
def downloadDataSet():
   cwd = os.getcwd()
  
   file_exists = exists('./dakshina_dataset_v1.0.tar')
   if(file_exists==False):
     print('downloading....')
     os.system('curl -SL https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar > dakshina_dataset_v1.0.tar')
     print('download Complete')
   extract_exists = exists('./dakshina_dataset_v1.0/')   
   if(extract_exists==False): 
     print('Extracting..') 
     with tarfile.open('dakshina_dataset_v1.0.tar', 'r') as archive:
         archive.extractall()
     print('Complete')
   print('You are all set')
def get_files(language):

  train_dir='./dakshina_dataset_v1.0/'+language+'/lexicons/'+language+'.translit.sampled.train.tsv'
  val_dir='./dakshina_dataset_v1.0/'+language+'/lexicons/'+language+'.translit.sampled.dev.tsv'
  test_dir='./dakshina_dataset_v1.0/'+language+'/lexicons/'+language+'.translit.sampled.test.tsv'
  
  return train_dir, val_dir, test_dir


"""Generate Tokens"""
def tokenize(lang,tokenizer=None):
    """ Uses tf.keras tokenizer to tokenize the data/words into characters
    """
    if(tokenizer==None):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(lang)
        lang_tensor = tokenizer.texts_to_sequences(lang)
        lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,
                                                            padding='post')
    else:
  
        lang_tensor = tokenizer.texts_to_sequences(lang)
        lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,
                                                        padding='post')

    return lang_tensor, tokenizer
def preprocess_data(fpath,ip_tokenizer=None, tgt_tokenizer=None):
   
    #Read data from files
    df = pd.read_csv(fpath, sep="\t", header=None)

    #Add start and end token
    df[0] = df[0].apply( lambda x:START_TOKEN+x+END_TOKEN) 
    ip_tensor, ip_tokenizer = tokenize(df[1].astype(str).tolist(), tokenizer=ip_tokenizer)
    
    tgt_tensor, tgt_tokenizer = tokenize(df[0].astype(str).tolist(), tokenizer=tgt_tokenizer) 
    
    dataset = tf.data.Dataset.from_tensor_slices((ip_tensor, tgt_tensor))
    dataset = dataset.shuffle(len(dataset))
    
    return dataset, ip_tokenizer, tgt_tokenizer



