import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:

    def __init__(self):
        self.embedding_size = None
        self.vectorizer = None

    def fit_embedding(self, corpus):
        raise NotImplementedError

    def get_embedding(self, corpus):
        raise NotImplementedError



class TF_IDF(Vectorizer):

    def __init__(self):
        self.embedding_size = 0
        self.vectorizer = TfidfVectorizer()

    def fit_embedding(self, corpus):
        return self.vectorizer.fit_transform(corpus)

    def get_embedding(self, corpus):
        return self.vectorizer.transform(corpus).toarray()

def read_preprocessed_data():
    files = glob.glob("data/pre_processed*")
    for file in files:
        vectorizer = TF_IDF()
        df = pd.read_csv(file)
        corpus = df["new"].values.astype('U')
        vectorizer.fit_embedding(corpus)

read_preprocessed_data()