import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd


training_data = pd.read_csv('intents_data.txt', '\t')
stemmer = LancasterStemmer()
corpus_words = {}
class_words = {}
classes = list(training_data['class'].unique())

for c in classes:
    class_words[c] = []

for i, data in training_data.iterrows():
    for word in nltk.word_tokenize(data['sentence']):
        if word not in ["?", "'s"]:
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            
            class_words[data['class']].extend([stemmed_word])

def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score += 1
            if show_details:
                print ("   match: %s" % stemmer.stem(word.lower() ))
    return score
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score += (1 / corpus_words[stemmer.stem(word.lower())])
            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score
def classify(sentence):
    high_class = None
    high_score = 0
    for c in class_words.keys():
        score = calculate_class_score_commonality(sentence, c, show_details=False)
        if score > high_score:
            high_class = c
            high_score = score
    return high_class, high_score
