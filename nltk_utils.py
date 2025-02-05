import nltk
#nltk.download('punkt_tab') #this is necessary to install the package so you need to run it only one time
from nltk.stem.porter import PorterStemmer  #this is to import a stemmer and there are different types
stemmer=PorterStemmer()

def tokenize(sentence):  #tokenizing is splitting the sentence into words
    return nltk.word_tokenize(sentence)

def stem(word):    #stemming is getting kinda the root of the word
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

a="hello there i am ali"
print(a)
a=tokenize(a)
print(a)

words=["organize","organizes","organizing"]
stemmed_words=[stem(w) for w in words]
print(stemmed_words)