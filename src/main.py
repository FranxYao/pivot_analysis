"""The Main Function 

Yao Fu, Columbia University
yao.fu@columbia.edu
Fri Jun 21st 2019 
"""

import argparse
import os
import numpy as np 

from config import Config
from data_utils import Dataset
from classifiers import LogisticClassifier, FFClassifier, CNNClassifier

def add_arguments(config):
  parser = argparse.ArgumentParser(description='Command line arguments')

  parser.add_argument('--dataset', type=str, default=config.dataset_name,
                      help='The dataset')
  parser.add_argument('--model', type=str, default=config.model,
                      help='The model')
  parser.add_argument('--set_to_test', type=str, default=config.set_to_test,
                      help='The model')
  parser.add_argument('--test_epoch', type=str, default=config.test_epoch,
                      help='The model')
  parser.add_argument('--bigram', action='store_true',
                      help='If use bigram feature')
  parser.add_argument('--trigram', action='store_true',
                      help='If use trigram feature')
  parser.add_argument('--vocab_cnt_thres', type=int, default=config.vocab_cnt_thres,
                      help='The occurrence threshold of the vocabulary') 
  parser.add_argument('--pivot_thres_cnt', type=float, default=config.pivot_thres_cnt,
                      help='The occurrence threshold of pivot words')              
  parser.add_argument('--prec_thres', type=float, default=config.prec_thres,
                      help='The threshold of precision')
  parser.add_argument('--recl_thres', type=float, default=config.recl_thres,
                      help='The threshold of recall')
  parser.add_argument('--filter_stop_words', type=int, default=config.filter_stop_words,
                      help='If use stop words')
  parser.add_argument('--max_training_case', type=int, default=-1,
    help='The maximum cases used in training the classifier') 
  parser.add_argument('--num_epoch', type=int, default=-1,
    help='The number of epoches') 
  parser.add_argument('--classifier', type=str, default=config.classifier,
                      help='The classifier')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='The index of gpu')
  return parser.parse_args()

def main():
  config = Config()
  args = add_arguments(config)
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

  # sample the dataset 
  config.parse_arg(args)
  dset = Dataset(config)
  print('------------------------------------------------------------')
  print('Pivot word discovery:')
  dset.build()
  config.vocab_size = len(dset.word2id)

  print('------------------------------------------------------------')
  print('Pivot classifier:')
  dset.classify()
  
  print('------------------------------------------------------------')
  print('Precision-recall histogram:')
  dset.get_prec_recl()

  print('------------------------------------------------------------')
  print('Storing the pivot outputs')
  dset.store_pivots()

  # the logistic classifier 
  if(args.classifier == 'ff'):
    classifier = FFClassifier(config)
    x_train, y_train = dset.to_bow_numpy('train')
    classifier.train(x_train, y_train)

    x_dev, y_dev = dset.to_bow_numpy('dev')
    classifier.test(x_dev, y_dev)

    x_test, y_test = dset.to_bow_numpy('test')
    classifier.test(x_test, y_test)
  elif(args.classifier == 'cnn'):
    cnn = CNNClassifier(config)
    x_train, y_train = dset.to_sent_numpy('train')
    cnn.train(x_train, y_train)

    x_dev, y_dev = dset.to_sent_numpy('dev')
    cnn.test(x_dev, y_dev)

    x_test, y_test = dset.to_sent_numpy('test')
    cnn.test(x_test, y_test)
  else:
    pass

  # correlation between the pivot words and logistic classifier words
  return 


if __name__ == '__main__':
  main()