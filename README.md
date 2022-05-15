# CS6910_assignment3
Report [wandb](https://wandb.ai/kankan-jana/CS6910_Assignment3/reports/CS6910-Assignment-3--VmlldzoyMDA4MDUw)
## Pre-requists 

**Python packages**	
Please install following packages in your local machine.
- pip install wandb
- pip install xtarfile
- pip install tqdm

**Guide to Execute Code**
# 
# Google Colab

 It is **recommanded** to run all the code in **Google Colab**.As most of the program require righ memory and GPU.Please upload the python notebook files available under [Note Book]For more tutorial related to Google colab usage follow the link: [Google Colab](https://colab.research.google.com/)
## IPython Notebook Links
 | Question  | Link  |Colab|
| --- | ----------- | ----------- | 
|Question 1,2,3,4 | [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_bestmodel.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XvRNNd4p-Vi4hc9LFXQc6SavUvsSPr69?usp=sharing)|
|Question 2(Vanilla sweep)| [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_Sweep.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9cEvnt8-6X37DSdTtd1ah0JAlS9BypG?usp=sharing)|
|Question 5(Attention Sweep)| [Sweep](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/assignment_3_With_attention_sweep.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CYzgZo3MS0qpi2fYousCdVcAAXRz7RGQ?usp=sharingg)|
|Question 6,7(Attention Best Model)| [Best Model](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_bestmodel_Attention.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ukUAwCJGfhbuqOAqFqDR0oxq1epsl8h6?usp=sharing)|
|Question 8(GPT)|  [GPT](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/gpt2_code.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OYdHwjafuuFax8KfG6zgV2Vzc2uIEb_-?usp=sharing)|


# Command Line

## Pre-requisite
It is optional to download and unzip dakshina_dataset datast before executing the programs.All program will automatically download these.
Links:
- Download the zip file from following link and place the file under parent directry 
[dakshina_dataset_v1]( https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
- Unzip the file in same directry

Run following files sequentially

## Main package



 | File  | Link  |
| --- | ----------- | 
|Seq2Seq  | [Seq2Seq.py](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/Seq2Seq.py)|


##  Run Sequence to Sequence Encoder Decoder
Command:

```
python Seq2Seq.py --language="te" --optimizer="adam" --lr="0.0005" --dropout="0.5" --inp_emb_size="64" --epoch="25" --cell_type="lstm" --num_of_encoders="1" --num_of_decoders="1" --patience="5" --batch_size="128" --latent_dim="256" --attention="False" --teacher_forcing_ratio="1" --save_outputs="save.csv"
```

| Param  | Accepted Values | Description|Default|
| --- | ----------- | ----------- |----------- |
|language *| ['bn' ,'gu','hi','kn','ml','mr','pa','sd','si','ta','te','ur']| Filter organization |'te'|
| optimizer | ['adam','rmsprop']  | Name of the optimizer| 'adam' |
| lr | Any Float value ex. 0.0001 |Learning Rates|0.0005|
| dropout | Any Float value ex. 0.5 |dropout Rates|0.5|
| inp_emb_size | Integer  | word embedding size |64|
| epoch |  number  |Epochs |25|
| cell_type |  ['lstm','gru','rnn']  |cell_type |lstm|
| num_of_encoders |  number  |total encoder | 1|
| num_of_decoders |  number  |total decoder | 1|
| patience |  number  |Early stopping condition | 5|
| batch_size |  number  |batch size | 128|
| latent_dim |  number  |latent dim | 256|
| attention |  Bool  |Apply attention layer | False|
| teacher_forcing_ratio |  Float  |Apply teacher forcing | 1|
| save_outputs |  string  |Save result of prediction| None|


# Prediction results


 | File  | Link  |
| --- | ----------- | 
|predictions_attention | [Link](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/predictions_attention.csv)|
|predictions_vanilla| [Link](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/predictions_vanilla.csv)|




# Questions 
- **Question 1,3,4 (Vanilla )** : Check the note book [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_bestmodel.ipynb).Alternatively ,
run the [Seq2Seq.py](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/Seq2Seq.py) python file following the instruction provided above.
In the ipython notebook we have couple of  observations has been used in Question 3 and 4. 

- **Question 2**: Check the note book [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_Sweep.ipynb) to run the sweep configuration.Upload the notebook in the google colab and run the file.Alternatively click the link [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9cEvnt8-6X37DSdTtd1ah0JAlS9BypG?usp=sharing)

- **Question 5 (Attention)** : Check the note book [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_bestmodel_Attention.ipynb).Please note ,In this note book ,we have shown numarious observations with attention model and also compare results with Vanilla model.So, it is recommanded to run this file instead the python file.
 In case some one want to execute python file please run run the [Seq2Seq.py](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/Seq2Seq.py) python file.Ensure to set parameter **--attention="True"**.
To run the sweep configuration for Attention, upload [Sweep](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/assignment_3_With_attention_sweep.ipynb) to google colab and execute it with High RAM configuration.Alternatively click  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CYzgZo3MS0qpi2fYousCdVcAAXRz7RGQ?usp=sharingg)

- **Question 6 (Connectivity)** : Excute the note book [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ukUAwCJGfhbuqOAqFqDR0oxq1epsl8h6?usp=sharing) .Please note this is same notebook used for question 5.

- **Question 8** : Open the notebook [GPT](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/gpt2_code.ipynb) or click here|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OYdHwjafuuFax8KfG6zgV2Vzc2uIEb_-?usp=sharing).Before execution , it is crucial to upload dataset **songs.txt**.Find this from here [Dataset](https://drive.google.com/file/d/1E0RGfMFvMUna6X9RFdBwAX2lR6g8bvBV/view?usp=sharing)
- Also find link of the trained model [Model](https://drive.google.com/file/d/1Xl4GbiWv_hz4za8MDO1zKh3pOsnJWZf1/view?usp=sharing)
