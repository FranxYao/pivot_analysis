"""Unified Dataset Utilities for Pivot Analysis

Yao Fu, Columbia University
yao.fu@columbia.edu
Fri Jun 21st 2019
"""

import pickle 
import time
import tqdm
import numpy as np 
from nltk.corpus import stopwords
from pivot_classifier import PivotClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical 


# STOPWORDS_PATH = '/home/francis/hdd/pivot_analysis/data/stop_words.pkl'
STOPWORDS_PATH = '../data/stop_words.pkl'
STOPWORDS = set(pickle.load(open(STOPWORDS_PATH, 'rb')))
STOPWORDS = STOPWORDS | set(stopwords.words('english'))

def upsample(sentences, label_id, target_size):
  """Upsample the sentences to the target size
  sample size = target size - original size
  Sample [sample size] sentences from the given sentence set. 
  Also modify the labels. 
  We assume all sentences are of the sample label

  Args:
    sentences: a list of the sentences to be upsampled
    label_id: the label of the sentences, an integer
    target_size: the size to sample to, an integer

  Returns:
    sampled_sentences: a list of sampled sentences
    labels: a list of extended labels 
  """
  sample_size = target_size - len(sentences)
  sample_id = np.random.choice(len(sentences), sample_size)

  sampled_sentences = list(sentences)
  for i in sample_id: sampled_sentences.append(sentences[i])
  labels = [label_id] * len(sampled_sentences)
  return sampled_sentences, labels

def downsample(sentences, label_id, target_size):
  """downsample the sentences to the target size
  Also modify the labels. 
  We assume all sentences are of the sample label

  Args:
    sentences: a list of the sentences to be upsampled
    label_id: the label of the sentences, an integer
    target_size: the size to sample to, an integer

  Returns:
    sampled_sentences: a list of sampled sentences
    labels: a list of extended labels 
  """
  sample_id = np.random.choice(len(sentences), target_size, False)

  sampled_sentences = []
  for i in sample_id: sampled_sentences.append(sentences[i])
  labels = [label_id] * len(sampled_sentences)
  return sampled_sentences, labels

def read_data(dataset, setname, base_path, balance_method='upsample'):
  """Read the data
  
  Args:
    dataset: the name of the dataset
    setname: 'train', 'dev' or 'test' 
    base_path: the base path of the datasets
    balance_method: 'upsample', 'downsample'
  
  Returns: 
    sentences: a list of sentences. A sentence is a list of words. 
    labels: a list of labels
  """
  print('Reading the %s dataset, %s .. ' % (dataset, setname))
  neg_path = base_path + dataset + '/' + setname + '.0'
  pos_path = base_path + dataset + '/' + setname + '.1'

  def _read_file_lines(f_path):
    with open(f_path, errors='ignore') as fd:
      lines = fd.readlines()
      lines_ = []
      for l in lines:
        s = l.split()
        if(len(s) > 0): lines_.append(s)
    return lines_

  neg_sentences = _read_file_lines(neg_path)
  pos_sentences = _read_file_lines(pos_path)

  neg_sent_num = len(neg_sentences)
  pos_sent_num = len(pos_sentences)
  print('neg sentence num: %d, pos num: %d' % (neg_sent_num, pos_sent_num))

  if(balance_method == 'upsample'):
    if(neg_sent_num < pos_sent_num): 
      neg_sentences, neg_labels = upsample(neg_sentences, 0, pos_sent_num)
      pos_labels = [1] * pos_sent_num
    else: 
      pos_sentences, pos_labels = upsample(pos_sentences, 1, neg_sent_num)
      neg_labels = [0] * neg_sent_num
  else: 
    if(neg_sent_num < pos_sent_num): 
      pos_sentences, pos_labels = downsample(pos_sentences, 1, neg_sent_num)
      neg_labels = [0] * neg_sent_num
    else: 
      neg_sentences, neg_labels = downsample(neg_sentences, 0, pos_sent_num)
      pos_labels = [1] * pos_sent_num

  sentences = neg_sentences + pos_sentences
  labels = np.array(neg_labels + pos_labels)
  return sentences, labels


def build_vocab(sentences, 
  is_bigram=False, is_trigram=False, cnt_threshold=5, filter_stop_words=1):
  """Build the vocabulary, bigram, and trigram
  
  Returns:
    unigram, bigram, trigram to id, and id to them

  Note: 
    the sentences is also padded here if its length is less than 3 
  """
  print("Building the vocabulary ..., filter_stop_words = %d" % filter_stop_words)
  start_time = time.time()
  unigram_cnt = dict()
  bigram_cnt = dict()
  trigram_cnt = dict()

  unigram2id = dict()
  bigram2id = dict()
  trigram2id = dict()
  id2unigram = dict()
  id2bigram = dict()
  id2trigram = dict()
  stop_words = STOPWORDS

  for s in sentences:
    slen = len(s)
    if(slen < 3):
      for i in range(3 - slen): s.append("_PAD")
      slen = 3
    for w in s:
      if(w not in unigram_cnt): unigram_cnt[w] = 1
      else: unigram_cnt[w] += 1
    if(is_bigram):
      for i in range(slen - 1):
        bigram = s[i] + " " + s[i + 1]
        if(bigram not in bigram_cnt): bigram_cnt[bigram] = 1
        else: bigram_cnt[bigram] += 1
    if(is_trigram):
      for i in range(slen - 2):
        trigram = s[i] + " " + s[i + 1] + " " + s[i + 2]
        if(trigram not in trigram_cnt): trigram_cnt[trigram] = 1
        else: trigram_cnt[trigram] += 1

  num_unigram = 2
  unigram2id['_PAD'] = 0
  id2unigram[0] = '_PAD'
  unigram2id['_UNK'] = 1
  id2unigram[1] = '_UNK'
  # num_unigram = 0
  for unigram in unigram_cnt:
    # if(filter_stop_words == 1 and unigram in stop_words): continue
    # if(unigram in stop_words): continue
    if(unigram == "_PAD"): continue
    if(unigram_cnt[unigram] < cnt_threshold): continue
    unigram2id[unigram] = num_unigram
    id2unigram[num_unigram] = unigram
    num_unigram += 1

  num_bigram = 0
  if(is_bigram):
    for bigram in bigram_cnt:
      if(bigram == "_PAD _PAD"): continue
      if(bigram_cnt[bigram] < cnt_threshold): continue
      bigram2id[bigram] = num_bigram
      id2bigram[num_bigram] = bigram
      num_bigram += 1
  else: bigram2id, id2bigram = None, None

  num_trigram = 0
  if(is_trigram):
    for trigram in trigram_cnt:
      if(trigram_cnt[trigram] < cnt_threshold): continue
      trigram2id[trigram] = num_trigram
      id2trigram[num_trigram] = trigram
      num_trigram += 1
  else: trigram2id, id2trigram = None, None 

  print("%d unigram, %d bigram, %d trigrams in total" % 
    (num_unigram, num_bigram, num_trigram))
  print("%.2s seconds cost" % (time.time() - start_time))
  return unigram2id, id2unigram, bigram2id, id2bigram, trigram2id, id2trigram

def build_style_word_sent(sentences, labels, unigram2id, bigram2id, trigram2id, 
  max_slen, id2style):
  """Build the style-word set, style-sentence set

  Args: 
    sentences: the sentence set, a list of sentences, 
      a sentence is a list of words
    labels: the labels, a list of integers
    unigram2id: word to index dictionary 
    bigram2id: bigram to index dictionary 
    trigram2id: trigram to index dictionary 
    max_slen: maximum sentence length. A sentence is a bag of words
    id2style: the index to style dictionary 
  
  Returns:
    style_sent_unigram: the style-sentence distribution. 
      A sentence is a bag of unigram 
    style_sent_bigram: the style-sentence(bigram) distribution
    style_sent_trigram: the style-sentence(trigram) distribution 
    style_unigram: the style-unigram distribution, a 2 * vocab_size matrix 
    style_bigram: the style-bigram distribution 
    style_trigram: the style-trigram distribution 
  """
  print("Building the style-related distributions ... ")

  start_time = time.time()

  style_sent_unigram = [[] for _ in id2style]
  style_sent_bigram = [[] for _ in id2style]
  style_sent_trigram = [[] for _ in id2style]

  style_unigram = np.zeros([len(id2style), len(unigram2id)])

  if(bigram2id is not None):
    style_bigram = np.zeros([len(id2style), len(bigram2id)])
  else: style_bigram = None

  if(trigram2id is not None):
    style_trigram = np.zeros([len(id2style), len(trigram2id)])
  else: style_trigram = None

  num_sentences = len(labels)

  for i in tqdm.tqdm(range(num_sentences)):
    s = sentences[i]
    lb = labels[i]

    s_unigram = []
    s_bigram = []
    s_trigram = []
    slen = len(s)
    for j in range(slen):
      w = s[j]
      if(w in unigram2id): 
        wid = unigram2id[w]
        s_unigram.append(wid)
        style_unigram[lb][wid] += 1
    s_unigram = set(s_unigram)

    if(bigram2id is not None):
      for j in range(slen - 1):
        bigram = s[j] + " " + s[j + 1]
        if(bigram in bigram2id): 
          bid = bigram2id[bigram]
          s_bigram.append(bid)
          style_bigram[lb][bid] += 1
      s_bigram = set(s_bigram)

    if(trigram2id is not None): 
      for j in range(slen - 2):
        trigram = s[j] + " " + s[j + 1] + " " + s[j + 2]
        if(trigram in trigram2id): 
          tid = trigram2id[trigram]
          s_trigram.append(tid)
          style_trigram[lb][tid] += 1
      s_trigram = set(s_trigram)

    style_sent_unigram[lb].append(s_unigram)
    style_sent_bigram[lb].append(s_bigram)
    style_sent_trigram[lb].append(s_trigram)
  
  # pad sentence bag-of-words to maximum length
  style_sent_unigram = _set_to_array(
    style_sent_unigram, num_sentences, max_slen, len(id2style))

  if(bigram2id is not None):
    style_sent_bigram = _set_to_array(
      style_sent_bigram, num_sentences, max_slen, len(id2style))
  else: style_sent_bigram = None

  if(trigram2id is not None):
    style_sent_trigram = _set_to_array(
      style_sent_trigram, num_sentences, max_slen, len(id2style))
  else: style_sent_trigram = None

  print("%.2s seconds cost" % (time.time() - start_time))
  # print(style_unigram[0][0], style_unigram[1][0])
  return (style_sent_unigram, style_sent_bigram, style_sent_trigram, 
    style_unigram, style_bigram, style_trigram)

def _set_to_array(style_sent, num_sentences, max_slen, num_style):
  sent_array = np.zeros([num_sentences, max_slen]).astype(np.int) - 1
  sid = 0
  for i in range(num_style):
    # print(len(style_sent[i]))
    for s in style_sent[i]:
      for wi, w in enumerate(s):
        sent_array[sid][wi] = w
        if(wi == max_slen - 1): break
      sid += 1
  assert(sid == num_sentences)
  return sent_array

def filter_dist(P, id2word, threshold=20):
  """filter the numbers less than [threshold] in the distribution and create the new 
  id2word dictionary, return the chosen index

  Args: 
    P: style-word the distribution 
    id2word: the index to word dictionary 
    threshold: the filter threshold

  Returns: 
    P_ret: the new style-word distribution
    id2word_new: the new id2word
    index_chosen: the chosen index 
  """
  index_chosen = np.unique(np.where(P > threshold)[1])
  rows = []
  columns = []
  for i in range(P.shape[0]):
    rows.append([i] * len(index_chosen))
    columns.append(index_chosen)
  P_ret = np.array(P[rows, columns])
  id2word_new = dict()
  filtered2prev = dict()
  for i, j in enumerate(index_chosen):
    id2word_new[i] = id2word[j]
    filtered2prev[i] = j
  return P_ret, id2word_new, index_chosen, filtered2prev

def prec_recl_f1_dist(style_words):
  """get the precision, recall, and f1 distribution of words
  
  Args:
    style_words: the vocabulary distribution given a style 

  Returns:
    prec_m: the precision matrix, prec_m[i][j] = what precision we can get if we 
      use word j to predict style i
    recl_m: the recall matrix, recl_m[i][j] = what recall we can get if we use 
      word j to predict style i
    f1_m: the f1 matrix, calculated as the harmonic mean of precision and recall
  """
  prec_m = np.zeros(style_words.shape)
  recl_m = np.zeros(style_words.shape)
  f1_m = np.zeros(style_words.shape)
  for i in range(style_words.shape[0]):
    for j in range(style_words.shape[1]):
      if(np.sum(style_words.T[j]) != 0):
        prec_m[i][j] = style_words[i][j] / np.sum(style_words.T[j])
      else: 
        prec_m[i][j] = 0
      if(np.sum(style_words[i]) != 0):
        recl_m[i][j] = style_words[i][j] / np.sum(style_words[i])
      else:
        recl_m[i][j] = 0
      f1_m[i][j] = 2 * prec_m[i][j] * recl_m[i][j]
      if(prec_m[i][j] + recl_m[i][j] != 0):
        f1_m[i][j] /= prec_m[i][j] + recl_m[i][j]
      else:
        f1_m[i][j] = 0
  return prec_m, recl_m, f1_m

def get_pivot_words(prec, recl, filtered2prev, stop_words, 
  prec_thres=0.7, recl_thres=0.):
  """Mine the high precision words

  Args: 
    prec: the precision matrix 
    recl: the recall matrix 
    filtered2prev: filtered index to previous word index mapping 
  """
  pivot_words = [[], []]
  pivot_prec_recl = [[], []]
  pivot_prec = [{}, {}]
  for si in range(2):
    words = np.where(prec[si] > prec_thres)
    total_recl = 0.
    for wi in words[0]:
      if(recl[si][wi] > recl_thres): 
        wid = filtered2prev[wi]
        if(wid in stop_words): continue
        pivot_words[si].append(wid)
        pivot_prec[si][wid] = prec[si][wi]
        # pivot_words[si].append(wi)
        pivot_prec_recl[si].append((prec[si][wi], recl[si][wi]))
        total_recl += recl[si][wi]
    print('class %d, %d pivots, pivot recall: %.4f' % 
      (si, len(pivot_words[si]), total_recl))
  return pivot_words, pivot_prec_recl, pivot_prec

def get_pivot_range(pivots, lower, upper):
  """Get the pivot words within a range"""
  p = set([w for w in pivots if pivots[w] >= lower and pivots[w] < upper])
  return p

class Dataset(object):
  def __init__(self, config):
    self.name = config.dataset_name
    self.show_statistics = config.show_statistics
    self.base_path = config.dataset_base_path
    self.is_bigram = config.is_bigram
    self.is_trigram = config.is_trigram
    self.max_slen = config.max_slen[self.name]
    self.style2id = config.style2id[self.name]
    self.id2style = config.id2style[self.name]
    self.threshold_cnt = config.pivot_thres_cnt
    self.prec_thres = config.prec_thres
    self.recl_thres = config.recl_thres
    self.vocab_cnt_thres = config.vocab_cnt_thres
    self.filter_stop_words = config.filter_stop_words
    self.max_training_case = config.max_training_case
    self.max_test_case = config.max_test_case

    self.output_path = config.output_path

    self.word2id = None
    self.id2word = None
    self.sentences = {'train': None, 'dev': None, 'test': None} 
    self.labels = {'train': None, 'dev': None, 'test': None}
    self.pivot_words = None
    self.pivot_prec = None
    self.pivot_words_prec_recl = None

    self.pivot_classifier = None
    self.prec_recl = None
    return 

  def build(self):
    """Build the dataset
    
    * read the sentences 
    * nlp-pipeline the sentences
    * build the word-class matrix
    * extract the pivot words
    * build the pivot classifier
    """
    
    ## Read the dataset
    balance_method = 'downsample' if self.name in ['reddit', 'twitter']\
      else 'upsample'
    train_sentences, train_labels = read_data(
      self.name, 'train', self.base_path, balance_method)
    dev_sentences, dev_labels = read_data(
      self.name, 'dev', self.base_path, balance_method)
    test_sentences, test_labels = read_data(
      self.name, 'test', self.base_path, balance_method)

    # Note: sentences are lists of words. Words are not converted to index at 
    # This stage 
    self.sentences = {
      'train': train_sentences, 'dev': dev_sentences, 'test': test_sentences}
    self.labels = {
      'train': train_labels, 'dev': dev_labels, 'test': test_labels}

    ## Dataset statistics
    if(self.show_statistics): pass # TBC 

    ## Piovt analysis pipeline 
    # word to index 
    (word2id, id2word, bigram2id, id2bigram, trigram2id, id2trigram) =\
      build_vocab(train_sentences, self.is_bigram, self.is_trigram, 
      self.vocab_cnt_thres, self.filter_stop_words)
    self.word2id, self.id2word = word2id, id2word

    # build style-word distribution 
    (style_sent_unigram, style_sent_bigram, style_sent_trigram, 
      style_words, style_bigram, style_trigram) = build_style_word_sent(
        train_sentences, train_labels, word2id, bigram2id, trigram2id, 
        self.max_slen, self.id2style)

    # filter words with small occurrance 
    style_words_filtered, id2word_filtered, index_chosen, filtered2prev = \
      filter_dist(style_words, id2word, self.threshold_cnt)

    # the precsion and recall of each words 
    prec, recl, f1 = prec_recl_f1_dist(style_words_filtered)
    self.prec = prec

    # pivot words are those with high precision 
    stop_words = set(word2id[w] for w in STOPWORDS if w in word2id) 
    self.pivot_words, self.pivot_words_prec_recl, self.pivot_prec =\
      get_pivot_words(
        prec, recl, filtered2prev, stop_words, self.prec_thres, self.recl_thres)

    self.pivot_classifier = PivotClassifier(
      self.pivot_words[1], self.pivot_words[0])
    return 

  def get_prec_recl(self, bins=10):
    """Get the precision-recall distribution"""
    prec_recl = np.zeros([2, bins])
    bin_range = 100. / bins
    for si in range(2):
      for j in range(bins):
        if(j < 5): continue
        lower = 0.01 * float(j) * bin_range
        upper = 0.01 * float(j + 1) * bin_range
        range_pivot_words = get_pivot_range(self.pivot_prec[si], lower, upper)
        pivot_recl = self.classify_w_pivot_list('train', range_pivot_words, si)
        print('class %d, prec lower %.3f, upper %.3f, %d pivot words, %.4f recall' 
          % (si, lower, upper, len(range_pivot_words), pivot_recl))
        
        print('%.4f recall' % pivot_recl)
        prec_recl[si][j] = pivot_recl

    self.prec_recl = prec_recl
    print('The precision-recall matrix:')
    print(self.prec_recl)

    out_path = self.output_path + self.name + '_prec_recl'
    print('Store to:')
    print(out_path)
    np.save(out_path, prec_recl)
    return prec_recl

  def store_pivots(self):
    """Store the pivot words, label sentences, and the precision-recall 
    histogram"""
    pivot_prec = {0: {}, 1: {}}
    for s in [0, 1]:
      out_path = self.output_path + self.name + '_%d.pivot' % s
      print('output stored in\n%s' % out_path)
      with open(out_path, 'w') as fd:
        for w, (p, r) in zip(
          self.pivot_words[s], self.pivot_words_prec_recl[s]):
          pivot_prec[s][w] = p
          fd.write('%s\t\t\t%.4f\t%.4f\n' % (self.id2word[w], p, r)) 
    
    # write down the sentences
    fd = {0: open(self.output_path + self.name + '_0.sent', 'w'),
          1: open(self.output_path + self.name + '_1.sent', 'w')}
    fd_hard = { 0: open(self.output_path + self.name + '_0.sent_hard', 'w'),
                1: open(self.output_path + self.name + '_1.sent_hard', 'w')}
    sent_out = [0, 0]
    print(np.sum(self.labels['dev'] == 0), np.sum(self.labels['dev'] == 1))
    for s, l in zip(self.sentences['dev'], self.labels['dev']):
      s_ = [self.word2id[w] if w in self.word2id else self.word2id['_UNK'] 
        for w in s]
      s_num_pivots = 0
      s_out = []
      for w, wid in zip(s, s_):
        if(wid in pivot_prec[l]):
          s_out.append(w + '(%.3f)' % pivot_prec[l][wid])
          s_num_pivots += 1
        else: s_out.append(w)
      if(s_num_pivots >= 3):
        fd[l].write(' '.join(s_out) + '\n') 
        sent_out[l] += 1
      if(s_num_pivots == 0):
        fd_hard[l].write(' '.join(s_out) + '\n')
    print('%d negative sentences written, %d positive' % 
      (sent_out[0], sent_out[1]))

    # Store the precision-recall histogram 
    # TBC 
    return 

  def classify_w_pivot_list(self, setname, pivot_list, s):
    """Classify the sentences with a given list of pivot words

    Args:
      setname: 'train', 'dev' or 'test'
      pivot_list: the list of pivot words
      s: the style label, 0 or 1 
    """
    recl = 0
    for i in range(len(self.sentences[setname])):
      x = self.sentences[setname][i]
      y = self.labels[setname][i]
      x = set(self.word2id[w] for w in x if w in self.word2id)
      if(len(set(x) & pivot_list) >= 1 and s == y): recl += 1
    recl = float(recl) / np.sum(self.labels[setname] == s)
    return recl

  def classify(self):
    """Test the pivot classifier"""
    self.pivot_classifier.test(
      self.sentences['train'], self.labels['train'], self.word2id, 'train')
    self.pivot_classifier.test(
      self.sentences['dev'], self.labels['dev'], self.word2id, 'dev')
    self.pivot_classifier.test(
      self.sentences['test'], self.labels['test'], self.word2id, 'test')
    return 

  def to_bow_numpy(self, setname):
    """Raw data to bag of words numpy representation"""
    if( (setname == 'train' and 
        len(self.sentences[setname]) > self.max_training_case) or 
        (setname in ['dev', 'test'] and 
        len(self.sentences[setname]) > self.max_test_case)):

      if(setname == 'train'): max_num_case = self.max_training_case
      else: max_num_case = self.max_test_case

      sample_id = np.random.choice(
        len(self.sentences[setname]), max_num_case, False)
      data = np.zeros([max_num_case, len(self.word2id)])
    else: 
      sample_id = range(len(self.sentences[setname]))
      data = np.zeros([len(self.sentences[setname]), len(self.word2id)])
    si = 0
    for sid in sample_id:
      s = self.sentences[setname][sid]
      for w in s:
        if w in self.word2id: 
          wid = self.word2id[w]
          data[si][wid] = 1
      si += 1
    labels = self.labels[setname][sample_id]
    return data, labels

  def to_sent_numpy(self, setname):
    """Raw data to numpy representation, a sentence is a list of word index"""

    if( (setname == 'train' and 
        len(self.sentences[setname]) > self.max_training_case) or 
        (setname in ['dev', 'test'] and 
        len(self.sentences[setname]) > self.max_test_case)):

      if(setname == 'train'): max_num_case = self.max_training_case
      else: max_num_case = self.max_test_case

      sample_id = np.random.choice(
        len(self.sentences[setname]), max_num_case, False)
    else: 
      sample_id = range(len(self.sentences[setname]))
    
    data = []
    for sid in sample_id:
      s = self.sentences[setname][sid]
      s_ = []
      for w in s:
        if w in self.word2id: wid = self.word2id[w]
        else: wid = self.word2id['_UNK']
        s_.append(wid)
      data.append(s_)
    data = pad_sequences(data, maxlen=self.max_slen)
    labels = self.labels[setname][sample_id]
    return data, labels

