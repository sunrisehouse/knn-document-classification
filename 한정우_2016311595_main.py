# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import csv
import numpy as np

from 한정우_2016311595_model import KNN
from 한정우_2016311595_model import Preprocessor

def main(args, logger):

    preprocessor = Preprocessor(args.data_dir, 'BBC_News_Data')
    x_train, y_train, x_test, y_test = preprocessor.preprocess()

    
    ### EDIT HERE ###
    config = {
        "student_id": '2016311595',
        "student_name": '한정우',

        "k": args.k,            # select K among 7, 9 or 11
        "metric": args.metric   # select distance metric among 'c' for cosine similarity or 'm' for manhattan
    }
    ### END ###

    model = KNN(logger, config)
    preds = model.predict(x_train, y_train, x_test)

    result = list()
    for p, l in zip(preds, y_test):
        if p == l:
            result.append(1)
        else:
            result.append(0)
    
    acc = np.sum(result) / len(result)

    with open(args.data_dir/'{}_{}_result.txt'.format(config['student_name'], config['student_id']), \
        'a', encoding='utf-8') as f:

        f.write("K:\t\t{}\n".format(str(config['k'])))
        f.write("Metric\t{}\n".format(config['metric']))
        f.write("Acc.\t\t{:.4f}\n\n".format(acc))
        
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=Path, default='./')
    parser.add_argument('--k',
                        type=int, default=9)
    parser.add_argument('--metric',
                        type=str, default='m')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args, logger)