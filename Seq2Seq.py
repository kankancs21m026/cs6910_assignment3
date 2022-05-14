START_TOKEN="\t"
END_TOKEN="\n"


import re, string
import numpy as np
import pandas as pd 
import os
from os.path import exists
import tensorflow as tf
import csv
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Import Custom packages
from Utility.Encoder  import Encoder
from Utility.Decoder  import Decoder
from Utility.Attention  import BahdanauAttention
from Utility.SeqTOSeq  import SequenceTOSequence
from Utility.Param  import Parameters
from Utility.DataLoader import downloadDataSet,get_files,tokenize,preprocess_data



#Get arguments
import argparse

#read arguments

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--dropout', type=float, required=False)
parser.add_argument('--language', type=str, required=False)

parser.add_argument('--teacher_forcing_ratio', type=float, required=False)
parser.add_argument('--inp_emb_size', type=int, required=False)
parser.add_argument('--epoch', type=int, required=False)
parser.add_argument('--cell_type', type=str, required=False)
parser.add_argument('--num_of_encoders', type=int, required=False)
parser.add_argument('--patience', type=int, required=False)
parser.add_argument('--num_of_decoders', type=int, required=False)
parser.add_argument('--batch_size', type=int, required=False)

parser.add_argument('--latent_dim', type=int, required=False)
parser.add_argument('--attention', type=bool, required=False)
parser.add_argument('--save_outputs', type=str, required=False)



args = parser.parse_args()

#Get arguments 

if(args.optimizer is None):
    optimizer='adam'
else:
    optimizer=args.optimizer


if(args.language is None):
    language="te"
else:
    language=args.language

if(args.num_of_encoders is None):
    num_of_encoders=1
else:
    num_of_encoders=args.num_of_encoders


if(args.batch_size is None):
    batch_size=128
else:
    batch_size=args.batch_size
if(args.inp_emb_size is None):
    inp_emb_size=64
else:
    inp_emb_size=args.inp_emb_size


if(args.cell_type is None):
    cell_type='lstm'
else:
    cell_type=args.cell_type



if(args.patience is None):
    patience=5
else:
    patience=args.patience

if(args.num_of_decoders is None):
    num_of_decoders=1
else:
    num_of_decoders=args.num_of_decoders


if(args.latent_dim is None):
    latent_dim=256
else:
    latent_dim=args.latent_dim

if(args.teacher_forcing_ratio is None):
    teacher_forcing_ratio=1
else:
    teacher_forcing_ratio=args.teacher_forcing_ratio

if(args.attention is None):
    attention=False
else:
    attention=args.attention


if(args.lr is None):
    lr=0.0005
else:
    lr=args.lr

if(args.epoch is None):
    epoch=5
else:
    epoch=args.epoch

if(args.dropout is None):
    dropout=0.5
else:
    dropout=args.dropout

# # Data Preprocessing

downloadDataSet()

#Get Token 

train_dir, val_dir, test_dir = get_files(language)

dataset, input_tokenizer, targ_tokenizer = preprocess_data(train_dir)
val_dataset, _, _ = preprocess_data(val_dir)

#train data 
dataset, input_tokenizer, targ_tokenizer = preprocess_data(train_dir)


## Train Model on Test Data

def run_model_on_test_dataset(model,param,test_dir):

    ## Character level accuracy ##
    test_dataset, _, _ = preprocess_data(test_dir, model.input_tokenizer, model.targ_tokenizer)
    test_loss, test_acc = model.evaluate(test_dataset, batch_size=100)
    print('Character level accuracy: '+str(test_acc.numpy()))

    ##  Word level accuracy ##
    test_tsv = pd.read_csv(test_dir, sep="\t", header=None)
    inputs = test_tsv[1].astype(str).tolist()
    targets = test_tsv[0].astype(str).tolist()
   
    outputs = []

    for word in inputs:
        outputs.append(model.translate(word)[0])

    print(f"Word level accuracy: {np.sum(np.asarray(outputs) == np.array(targets)) / len(outputs)}")

    if param.save_outputs is not None:
        df = pd.DataFrame()
        df["inputs"] = inputs
        df["targets"] = targets
        df["outputs"] = outputs
        df.to_csv(param.save_outputs)


    return model

param=Parameters(language="te",\
        embedding_dim=inp_emb_size,\
        encoder_layers=num_of_encoders,\
        decoder_layers=num_of_decoders,\
        layer_type=cell_type,\
        units=latent_dim,\
        dropout=dropout,\
        epochs=epoch,\
        batch_size=batch_size,\
        attention=attention\
        )
param.patience=patience
param.save_outputs=save_outputs
#Define model
model = SequenceTOSequence(param)
model.set_vocabulary(input_tokenizer, targ_tokenizer)
model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\
            metric = tf.keras.metrics.SparseCategoricalAccuracy(),\
            optimizer = optimizer,\
            lr=lr\
            )
param.teacher_forcing_ratio=teacher_forcing_ratio
model.fit(dataset, val_dataset, epochs=param.epochs, wandb=None, teacher_forcing_ratio=param.teacher_forcing_ratio)                  


#Evaluate model
run_model_on_test_dataset(model,param,test_dir)