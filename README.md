# CS6910_assignment3

## Pre-requists 

**Python packages**	
Please install following packages in your local machine.
- pip install wandb
- pip install xtarfile


**Guide to Execute Code**
# 
# Google Colab

 It is **recommanded** to run all the code in **Google Colab**.As most of the program require righ memory and GPU.Please upload the python notebook files available under [Note Book]For more tutorial related to Google colab usage follow the link: [Google Colab](https://colab.research.google.com/)
## IPython Notebook Links
 | Question  | Link  |Colab|
| --- | ----------- | ----------- | 
|Question 1,2,3,4 | [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_bestmodel.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XvRNNd4p-Vi4hc9LFXQc6SavUvsSPr69?usp=sharing)|
|Question 2(sweep)| [NoteBook](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/DL_Assignment3_Sweep.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9cEvnt8-6X37DSdTtd1ah0JAlS9BypG?usp=sharing)|
|Question 5(Attention)| [Sweep](https://github.com/kankancs21m026/cs6910_assignment3/blob/main/assignment_3_With_attention_sweep.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CYzgZo3MS0qpi2fYousCdVcAAXRz7RGQ?usp=sharingg)|
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
|CNN  | [CNN.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/utility/CNN.py)|
|Import Dataset | [Dataset.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/utility/Dataset.py)|



 | Question  | Link  |
| --- | ----------- | 
|Question 1| [PartA_Question1_CNN.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_Question1_CNN.py)|
|Question 1,2,3| [PartA_Question2_Sweep.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_Question2_Sweep.py)|
|Question 4,5| [PartA_parta_qestion_4_5.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_parta_qestion_4_5.py)|


##  Question 1
Command:

python PartA_Question1_CNN.py 

| Param  | Accepted Values | Description|Required|
| --- | ----------- | ----------- |----------- |
|filterOrganization *| ['config_all_same','config_incr','config_decr','config_alt_incr','config_alt_decr','Custom']| Filter organization |Yes|
| no_of_filters |  Comma delimited list input Example "64,64,64,64,64"  | Name of the optimizer| Only when filterOrganization selected as **custom** |
| optimizer |  [Adam,sgd]  | Name of the optimizer|No|
| lr | Any Float value ex. 0.0001 |Learning Rates|No|
| dropout | Any Float value ex. 0.0001 |dropout Rates|No|
| image_size |  integer  |size of the image |No|
| batchNormalization |  Bool  |Batch Normalisation |No|
| augment_data |  Bool  |Preprocess data |No|
| number_of_neurons_in_the_dense_layer |  integer  |size of the dense layer |No|
| activation_function| string | Activation function|No|
| wandbLog |  Bool  |Log in Wandb  |No|
* filterOrganization
- **config_all_same** : [64,64,64,64,64]
- **config_incr** : [16,32,64,128,256]
- **config_decr** : [256,128,64,32,16]
- **config_alt_incr** : [32,64,32,64,32]
- **config_alt_decr** : [64,32,64,32,64]
- **Custom** :  Any custome configuration as input.Respective parameters **no_of_filters**

Example
Please use any of the command and change respective parameters **no_of_filters** is optional parameter

**filterOrganization as "config_all_same"** Please note in this case 
```
python PartA_Question1_CNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "config_all_same" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" 
```

**filterOrganization as "custom"**
```
python PartA_Question1_CNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "custom" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" --no_of_filters "64,64,64,64,64"
```



##  Question 2
Running sweep configuration
```
python PartA_Question2_Sweep.py
```



##  Question 4,5
Run the best model 

### Pre-requisite

In case there is no file **model-best.h5** in PARTA directry , please run the following command
```
PartA_parta_qestion_4_BestModel.py
```
**Alternatively** please download the model from below link 
https://drive.google.com/file/d/1bdMa03-Jf-zlZi1zL1IQLGThvNfAfHAz/view?usp=sharing

View the run in wanDb:
https://wandb.ai/kankan-jana/CS6910_Assignment-2/runs/179siwiu/files?workspace=user-kankan-jana

Download the model (image given below)

![image](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/image/wandb.jpg)

After that  run below set of commands
#### Question 4

```
python PartA_parta_qestion_4.py
```
#### Question 5

```
python PartA_parta_qestion_5.py
```
