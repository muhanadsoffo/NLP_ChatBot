import nltk
import numpy as np
#nltk.download('punkt_tab') #this is necessary to install the package so you need to run it only one time
from nltk.stem.porter import PorterStemmer  #this is to import a stemmer and there are different types
stemmer=PorterStemmer()

def tokenize(sentence):  #tokenizing is splitting the sentence into words
    return nltk.word_tokenize(sentence)

def stem(word):    #stemming is getting kinda the root of the word
    return stemmer.stem(word.lower())

#this function converts a sentence into a vector in numerical representation
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag= np.zeros(len(all_words), dtype=np.float32) # Creates an empty vector with size of vocabulary
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0

    return bag