import csv
import random

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from tqdm import tqdm

import math

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
        print('CAL TFIDF START : ', data[0])
        dataLength = len(data)

        tokenKeyList = []
        for document in data:
            for token in document:
                hasToken = False

                for tokenKey in tokenKeyList:
                    if (tokenKey == token):
                        hasToken = True
                        break
                
                if hasToken == False:
                    tokenKeyList.append(token)
        print('TOKENKEY LIST DONE : ', tokenKeyList)

        tf = []
        documentIdx = 0
        for document in data:
            tf.append([])

            tokenKeyIdx = 0
            for tokenKey in tokenKeyList:
                tf[documentIdx].append(0)
                for token in document:
                    if tokenKey == token:
                        tf[documentIdx][tokenKeyIdx] += 1
                tokenKeyIdx += 1

            documentIdx += 1
        print('TF DONE : ', tf[0])

        df = []
        tokenKeyIdx = 0
        for tokenKey in tokenKeyList:
            df.append(0)
            for document in data:
                isFounded = False
                for token in document:
                    if token == tokenKey:
                        isFounded = True
                        break
                if isFounded == True:
                    df[tokenKeyIdx] += 1
            tokenKeyIdx += 1
        print('DF DONE : ', df)

        idf = []
        for dfValue in df:
            idfValue = math.log(dataLength / (1 + dfValue))
            idf.append(idfValue)

        print('IDF DONE : ', idf)
        tfidfIdx = 0
        for tfDocument in tf:
            tfidf.append([])
            for tfValue, idfValue in zip(tfDocument, idf):
                tfidf[tfidfIdx].append(tfValue*idfValue)

            tfidfIdx += 1

        print('TFIDF DONE : ', tfidf[0])
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
        for testTfidfValue in test_data:
            neighborIdx = 0
            labelNearList = []
            for neighborTfidfValue in neighbors:
                neighborLabel = labels[neighborIdx]
                neighborNear = self.nearness(testTfidfValue, neighborTfidfValue)

                foundedIndex = -1
                labelNearIndex = 0
                for labelNear in labelNearList:
                    if labelNearIndex >= self.k:
                        break
                    
                    curNear = labelNear[1]

                    if self.metric == 'c':
                        if curNear < neighborNear:
                            foundedIndex = labelNearIndex
                            break
                    elif self.metric == 'm':
                        if curNear > neighborNear:
                            foundedIndex = labelNearIndex
                            break
                    
                    labelNearIndex += 1

                if foundedIndex != -1:
                    labelNearList.insert(foundedIndex, (neighborLabel, neighborNear))
                else:
                    labelNearList.append((neighborLabel, neighborNear))
                if len(labelNearList) > self.k:
                    labelNearList.pop()

                neighborIdx += 1

            labelCountDict = {}
            for labelNear in labelNearList:
                label = labelNear[0]

                if label not in labelCountDict:
                    labelCountDict[label] = 0
                
                labelCountDict[label] += 1

            labelCountList = [(k,v) for k,v in labelCountDict.items()]

            maxCount = 0
            maxIdx = -1
            labelIdx = 0

            for (label, count) in labelCountList:
                if maxCount < count:
                    maxIdx = labelIdx
                    maxCount = count
                labelIdx += 1

            maxLabel = ''
            if maxIdx != -1:
                maxLabel = labelCountList[maxIdx][0]
            print(maxLabel, ' : ',labelNearList)
            pred.append(maxLabel)


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
            value1 = 0
            value2 = 0
            value3 = 0
            
            for xi, yi in zip(x, y):
                value1 += xi * yi

                value2 += xi * xi

                value3 += yi * yi

            near = value1 / (math.sqrt(value2) * math.sqrt(value3))
            pass
        elif self.metric == 'm':
            near = 0
            for xi, yi in zip(x, y):
                value1 = xi - yi

                if value1 < 0:
                    value1 *= -1
                
                near += value1
            pass
        else:
            self.logger.info("Unknown Metric")

        ### END ###

        return near