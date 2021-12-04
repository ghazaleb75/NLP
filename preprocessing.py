from __future__ import unicode_literals
from hazm import *
import csv
import pandas as pd
from hazm.utils import stopwords_list
lemmatizer = Lemmatizer()
normalizer = Normalizer()

def preprocess(sent):
    sent = sent.replace("؟!.،,?" ,"")
    normalized = normalizer.normalize(sent)
    words = word_tokenize(normalized)
    stop_words = stopwords_list()
    word_list = []
    for word in words:
        if word in stop_words:
            continue
        res = lemmatizer.lemmatize(word)
        if '#' in res:
            res = res.split("#")[1]
        word_list.append(res)
    return word_list


def read_file(file_name):

    pairs = []
    with open('data/%s.csv'%file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sent = row[1]
            res = preprocess(sent)
            new_sent = ' '.join(res)
            pairs.append((row[1], new_sent, row[2]))
    df = pd.DataFrame(pairs, columns = ['old','new', 'tag'])
    df.to_csv('pre_processed_%s.csv'%file_name)
