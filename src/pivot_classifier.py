"""The pivot classifier

Yao Fu, Columbia University 
yao.fu@columbia.edu
Tue Jun 18, 2019 
"""

import numpy as np

from sklearn.metrics import classification_report
from tqdm import tqdm
from pprint import pprint

class PivotClassifier(object):

  def __init__(self, pos_index, neg_index):
    """
    Args: 
      pos_index: the list of positive word index 
      neg_index: the list of negative word index
    """
    # print('pos index:')
    # print(pos_index[:20])
    # print('neg index:')
    # print(neg_index[:20])
    self.pos = set(pos_index)
    self.neg = set(neg_index)
    return 

  def classify(self, sent):
    """Classify a sentence
    
    Args:
      sent: a list of word index
    """
    sent_ = set(sent)
    # print(sent_)
    pos_cnt = len(self.pos & sent_)
    neg_cnt = len(self.neg & sent_)
    # print(pos_cnt, neg_cnt, pos_cnt > neg_cnt)
    if(pos_cnt >= neg_cnt): 
      # p = float(pos_cnt) / len(sent_)
      p = 0
      return 1, p
    else: 
      # p = float(neg_cnt) / len(sent_)
      p = 0
      return 0, p

  def classify_dataset(self, sentences, word2id):
    """classify a dataset
    
    Args: 
      sentences: the list of sentences, a sentence is a list of words 

    Returns:  
      outputs: a numpy array with value 0=negative, 1=positive
    """
    outputs = []
    # for s in tqdm(sentences): 
    for s in sentences:
      s_ = []
      for w in s:
        if(w in word2id): s_.append(word2id[w])
      outputs.append(self.classify(s_)[0])
    return np.array(outputs)

  def test(self, sentences, labels, word2id, setname):
    """Classify a dataset and measure the performance
    
    Args: 
      sentences: the list of sentences, a sentence is a list of word index 
      labels: the list of labels
    """
    labels = np.array(labels)
    pred = self.classify_dataset(sentences, word2id)
    print(np.sum(pred == 0))
    print(np.sum(pred == 1))

    # performance 
    # results = classification_report(labels, pred, output_dict=True)
    # pprint(results)
    results = np.sum(pred == labels) / float(len(labels))
    print('%s accuracy: %.4f' % (setname, results))
    return results