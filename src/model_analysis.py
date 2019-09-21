"""Model output analysis

Yao Fu, Columbia University
yao.fu@columbia.edu
Sat Jul 06th 2019 
"""

import argparse
import os
import numpy as np 

from config import Config
from data_utils import Dataset
from main import add_arguments
from collections import Counter
from editdistance import eval as editdist

def _sent_raw_to_id(sentences, word2id, setname):
  sentences_ = [] 
  num_unk = 0
  total_words = 0
  unknown_words = []
  for s in sentences:
    s_ = []
    for w in s:
      if(w in word2id): s_.append(word2id[w])
      else:
        num_unk += 1
        s_.append(word2id['_UNK'])
        unknown_words.append(w)
      total_words += 1
    sentences_.append(s_)
  ratio = float(num_unk) / total_words
  print('%s,%d unk words, %d total, %.4f ratio' % 
    (setname, num_unk, total_words, ratio))
  unknown_words = Counter(unknown_words)
  with open(setname + '_unknown_words.txt', 'w') as fd:
    for w, c in unknown_words.most_common(): fd.write('%s %d\n' % (w, c))
  return sentences_

def _transfer_in_pivot(src, tsf, pivots):
  total_modified = []
  total_in_pivot = []
  sent_lens = []
  for s, t in zip(src, tsf):
    s_ = set(s)
    t_ = set(t)
    sent_lens.append(len(s))
    modified = (s_ | t_) - (s_ & t_)
    num_modified = len(modified)
    num_in_pivot = len(modified & pivots)
    total_modified.append(num_modified)
    total_in_pivot.append(num_in_pivot)
  total_modified = np.sum(total_modified)
  total_in_pivot = np.sum(total_in_pivot)
  avg_modified = total_modified / float(2 * len(src))
  ratio = float(total_in_pivot) / total_modified
  avg_sent_lens = np.average(sent_lens)
  return total_modified, total_in_pivot, avg_modified, ratio, avg_sent_lens

def _format_sentence(s, id2word, pivots):
  s_ = []
  for w in s:
    if(w in pivots[0]): s_.append('[0 ' + id2word[w] + ']')
    elif(w in pivots[1]): s_.append('[1 ' + id2word[w] + ']')
    else: 
      # print(type(w))
      s_.append(id2word[w])
  return ' '.join(s_)

def _masked_edit_dist(src, tsf, pivots, id2word, output_path):
  distances = []
  i = 0
  print('output write to:\n%s' % output_path)
  # print(output_path)
  fd = open(output_path, 'w')
  for s, t in zip(src, tsf):
    s_ = set(s)
    t_ = set(t)
    modified = (s_ | t_) - (s_ & t_)
    # s_masked = [w if w not in modified else 0 for w in s] # 0 = '_PAD'
    # t_masked = [w if w not in modified else 0 for w in t] # 0 = '_PAD'

    pivot_set = pivots[0] | pivots[1]
    s_masked = [w if w not in pivot_set else 0 for w in s] # 0 = '_PAD'
    t_masked = [w if w not in pivot_set else 0 for w in t] # 0 = '_PAD'
    s_masked_ = ' '.join([str(w) for w in s_masked])
    t_masked_ = ' '.join([str(w) for w in t_masked])
    ed = editdist(s_masked_, t_masked_)
    distances.append(ed)
    
    fd.write('s: %s\n' % _format_sentence(s, id2word, pivots))
    fd.write('t: %s\n' % _format_sentence(t, id2word, pivots))
    # debug
    # if(i < 5):
    #   print('modified:', [id2word[w] for w in modified])
    #   print('s:', _format_sentence(s, id2word, pivots))
    #   print('t:', _format_sentence(t, id2word, pivots))
    #   print('s_masked:', _format_sentence(s_masked, id2word, pivots))
    #   print('t_masked:', _format_sentence(t_masked, id2word, pivots))
    #   print('ed %d' % ed)
    #   i += 1
  avg_dist = np.average(distances)
  distances = Counter(distances)

  dist_distribution = np.zeros(8)
  for i in range(8): 
    if(i < 7): dist_distribution[i] = float(distances[i]) / len(src)
    else: dist_distribution[i] = 1 - dist_distribution[: i].sum()
  return avg_dist, distances, dist_distribution

class PivotTransferAnalysis(object):
  """Pivot analysis of the transfered dataset"""

  def __init__(self, config):
    self.data_base_path = config.dataset_base_path + config.dataset_name +\
      '_transfer/' + config.model + '/' + config.set_to_test + '.'
    if(config.model == 'cmu'):
      if(config.test_epoch != ''): 
        self.data_base_path = self.data_base_path + config.test_epoch + '.'
    if(config.model == 'mit'):
      self.data_base_path_tsf = self.data_base_path + 'epoch' + config.test_epoch + '.'
    self.output_path = config.output_path + 'transfer/' + config.model +\
      '_' + config.dataset_name + '_transfer'
    return 

  def pipeline_w_cmu(self, dset):
    src = open(self.data_base_path + 'src').readlines()
    src = [s.split() for s in src]
    src = _sent_raw_to_id(src, dset.word2id, 'src')

    tsf = open(self.data_base_path + 'tsf').readlines()
    tsf = [s.split() for s in tsf]
    tsf = _sent_raw_to_id(tsf, dset.word2id, 'tsf')

    pivot_words = set(dset.pivot_words[0]) | set(dset.pivot_words[1])

    modified, in_pivot, avg_modified, ratio, avg_sent_len =\
      _transfer_in_pivot(src, tsf, pivot_words)
    print('%d modified, %d in pivot' % (modified, in_pivot))
    print('%.2f avg sentence length %.2f average modified, %.4f ratio' % 
      (avg_sent_len, avg_modified, ratio))

    pivot_words_class = [set(dset.pivot_words[0]), set(dset.pivot_words[1])]
    avg_dist, distances, dist_distribution =\
      _masked_edit_dist(src, tsf, pivot_words_class, dset.id2word, self.output_path)
    print('%d different distances in total, avg %.2f' % 
      (len(distances), avg_dist))
    print('distribution:', np.sum(dist_distribution))
    for i, di in enumerate(dist_distribution):
      print('%d: %.4f' % (i, di))  
    print(distances.most_common(10))
    return 

  def pipeline(self, dset):
    """Pivot analysis pipeline """
    print('reading data from:\n  %s' % self.data_base_path)
    # Read the transfered sentences
    neg_src = open(self.data_base_path + '0.src').readlines()
    neg_src = [s.split() for s in neg_src]
    neg_src = _sent_raw_to_id(neg_src, dset.word2id, 'neg_src')

    neg_tsf = open(self.data_base_path_tsf + '0.tsf').readlines()
    neg_tsf = [s.split() for s in neg_tsf]
    neg_tsf = _sent_raw_to_id(neg_tsf, dset.word2id, 'neg_tsf')

    pos_src = open(self.data_base_path + '1.src').readlines()
    pos_src = [s.split() for s in pos_src]
    pos_src = _sent_raw_to_id(pos_src, dset.word2id, 'pos_src')

    pos_tsf = open(self.data_base_path_tsf + '1.tsf').readlines()
    pos_tsf = [s.split() for s in pos_tsf]
    pos_tsf = _sent_raw_to_id(pos_tsf, dset.word2id, 'pos_tsf')

    # calculate how many modified words are pivots
    pivot_words = set(dset.pivot_words[0]) | set(dset.pivot_words[1])

    # neg_modified, neg_in_pivot, neg_avg_modified, neg_ratio =\
    #   _transfer_in_pivot(neg_src, neg_tsf, pivot_words)
    # print('neg to pos, %d modified, %d in pivot' % (neg_modified, neg_in_pivot))
    # print('%.2f average modified, %.4f ratio' % (neg_avg_modified, neg_ratio))

    # pos_modified, pos_in_pivot, pos_avg_modified, pos_ratio =\
    #   _transfer_in_pivot(pos_src, pos_tsf, pivot_words)
    # print('pos to neg, %d modified, %d in pivot' % (pos_modified, pos_in_pivot))
    # print('%.2f average modified, %.4f ratio' % (pos_avg_modified, pos_ratio))

    modified, in_pivot, avg_modified, ratio, avg_sent_lens =\
      _transfer_in_pivot(neg_src + pos_src, neg_tsf + pos_tsf, pivot_words)
    print('%d modified, %d in pivot' % (modified, in_pivot))
    print('%.2f avg len, %.2f average modified, %.4f in pivots' % 
      (avg_sent_lens, avg_modified, ratio))

    # mask the modified words, calculate the sentence distances 
    pivot_words_class = [set(dset.pivot_words[0]), set(dset.pivot_words[1])]
    # avg_dist, distances, _ = _masked_edit_dist(
    #   neg_src, neg_tsf, pivot_words_class, dset.id2word, self.output_path)
    # print('neg to pos, %d different distances in total, avg %.2f' % 
    #   (len(distances), avg_dist))
    # print(distances.most_common(10))

    # avg_dist, distances, _ = _masked_edit_dist(
    #   pos_src, pos_tsf, pivot_words_class, dset.id2word, self.output_path)
    # print('pos to neg, %d different distances in total, avg %.2f' % 
    #   (len(distances), avg_dist))
    # print(distances.most_common(10))

    avg_dist, distances, dist_distribution = _masked_edit_dist(
      neg_src + pos_src, neg_tsf + pos_tsf, pivot_words_class, dset.id2word, self.output_path)
    print('%d different distances in total, avg %.2f' % 
      (len(distances), avg_dist))
    print('distribution:', np.sum(dist_distribution))
    for i, di in enumerate(dist_distribution):
      print('%d: %.4f' % (i, di))  
    print(distances.most_common(10)) 

    # mask the pivot words, calculate the sentence distances
    return 

def main():
  config = Config()
  args = add_arguments(config)
  config.parse_arg(args)
  dset = Dataset(config)
  dset.build()
  # print('debug:')
  # print(dset.id2word[1])
  config.vocab_size = len(dset.word2id)

  # read the transfered sentences
  transfer_analysis = PivotTransferAnalysis(config)

  if(config.model == 'cmu'):
    transfer_analysis.pipeline_w_cmu(dset)
  else:
    transfer_analysis.pipeline(dset)
  return 

if __name__ == '__main__':
  main()