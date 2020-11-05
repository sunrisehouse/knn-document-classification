import csv
import random

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from tqdm import tqdm

nltk.download('punkt')                      
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class Preprocessor:

    def __init__(self, data_dir, data_name):
        
        self.text = list()
        self.label = list()

        with open(data_dir/'{}.csv'.format(data_name), 'r', encoding='utf-8-sig') as f:
            rdr = csv.reader(f)

            for line in rdr:
                self.text.append(line[1])
                self.label.append(line[2])
        
        self.text = self.text[1:]
        self.label = self.label[1:]
    
    def preprocess(self):

        tokens = self.tokenize()
        tf_idf = self.cal_tf_idf(tokens)

        dataset = list()
        for v, l in zip(tf_idf, self.label):
            dataset.append([v, l])
        
        random.shuffle(dataset)
        train, test = self.split_data(dataset)

        x_train = list()
        y_train = list()
        for v, l in train:
            x_train.append(v)
            y_train.append(l)
        
        x_test = list()
        y_test = list()
        for v, l in test:
            x_test.append(v)
            y_test.append(l)
        
        return x_train, y_train, x_test, y_test

    def split_data(self, data):

        train = data[:int(len(data) * 0.8)]
        test = data[int(len(data) * 0.8):]

        return train, test

    def tokenize(self):
        
        tags =  ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        stop_words = set(stopwords.words('english'))
        tokenized_text = list()

        for txt in tqdm(self.text, desc="tokenizing..."):
            txt = txt.rstrip("\n")
            txt = txt.lower()

            pos_tagged = nltk.pos_tag(word_tokenize(txt))
            tokenized_text.append(
                [token[0] for token in pos_tagged \
                    if token[1] in tags \
                        if not token[0] in stop_words]
            )

        return tokenized_text
    
    def cal_tf_idf(self, data):
        """
        Function calculating tf-idf

        Parameter
            data: tokenized text list.

        Return
            tfidf vector of the data
        """
        tfidf = list()

        ### EDIT HERE ###

        ### END ###

        return tfidf


class KNN:

    def __init__(self, logger, config):

        self.logger = logger

        self.metric = config['metric']
        self.k = config['k']
        
    def predict(self, neighbors, labels, test_data):
        """
        Function predicting label of data.

        Parameter
            neighbors: tf-idf vectors of articles.
            labels: label of neighbors.
            test_data: data that you have to assign a label.

        Return
            pred: predictions for the test data.
        """
        pred = list()

        ### EDIT HERE ###

        ### END ###

        return pred

    def nearness(self, x, y):
        """
        Function calculating distance between two vectors.

        Parameter
            x: a vector consisting of tf-idf values.
            y: a vector consisting of tf-idf values.

        Return
            metric: distance or similarity between two data points.
        """
        near = None

        ### EDIT HERE ###

        if self.metric == 'c':
            pass
        elif self.metric == 'm':
            pass
        else:
            self.logger.info("Unknown Metric")

        ### END ###

        return near
