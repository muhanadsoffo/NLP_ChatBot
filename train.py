import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize,stem,bag_of_words
with open('intents.json','r') as f:
    intents =json.load(f)

all_words= []
tags=[]
xy=[]

for intent in intents['intents']:
    tag= intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)  #because it's an array we don't use append
        xy.append((w,tag))   # Store pattern with corresponding tag


ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words= sorted(set(all_words)) #here we are removing the duplicated words
x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
    bag= bag_of_words(pattern_sentence,all_words)  # Convert pattern to numerical bag-of-words
    x_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

x_train=np.array(x_train)
y_train=np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train


    def __getitem__(self, index): #Returns one training example (features + label).
        return self.x_data[idx] , self.y_data[idx]
    
    def __len__(self): #Returns total number of training examples.
        return self.n_samples
    

#hyperparameters
batch_size=8

dataset=ChatDataset()
train_loader=DataLoader (dataset=dataset, batch_size=batch_size, shuffle=True ,num_workers=2)