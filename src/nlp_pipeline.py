"""The NLP data cleaning pipeline
Yao Fu, Columbia University 
yao.fu@columabia.edu
THU MAY 09TH 2019 
"""
import numpy as np 

import nltk 
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm 

def normalize(
  sentences, word2id, start, end, unk, pad, max_src_len, max_tgt_len):
  """Normalize the sentences by the following procedure
  - word to index 
  - add unk
  - add start, end
  - pad/ cut the sentence length
  - record the sentence length
  Returns: 
    sent_normalized: the normalized sentences, a list of (src, tgt) pairs
    sent_lens: the sentence length, a list of (src_len, tgt_len) pairs
  """
  sent_normalized, sent_lens = [], []

  def _pad(s, max_len, pad):
    s_ = list(s[: max_len])
    lens = len(s_)
    for i in range(max_len - lens):
      s_.append(pad)
    return s_

  for (s, t) in tqdm(sentences):
    s_ = []
    s_.extend([word2id[w] if w in word2id else unk for w in s])
    s_.append(end)
    slen = min(len(s) + 1, max_src_len)
    s_ = _pad(s_, max_src_len, pad)

    t_ = [start]
    t_.extend([word2id[w] if w in word2id else unk for w in t])
    t_.append(end)
    tlen = min(len(t) + 1, max_tgt_len)
    t_ = _pad(t_, max_tgt_len, pad)

    sent_normalized.append((s_, t_))
    sent_lens.append((slen, tlen))

  return sent_normalized, sent_lens

def corpus_statistics(sentences, vocab_size_threshold=5):
  """Calculate basic corpus statistics"""
  print("Calculating basic corpus statistics .. ")

  # sentence length
  sentence_lens = []
  for s in sentences: sentence_lens.append(len(s))
  sent_len_percentile = np.percentile(sentence_lens, [50, 80, 90, 95, 100])
  print("sentence length percentile:")
  for i, percent in enumerate([50, 80, 90, 95, 100]):
    print('%d: %d' % (percent, sent_len_percentile[i]))

  # vocabulary
  vocab = []
  for s in sentences:
    vocab.extend(s)
  vocab = Counter(vocab)
  print("vocabulary size: %d" % len(vocab))
  for th in range(1, vocab_size_threshold + 1):
    vocab_truncate = [w for w in vocab if vocab[w] >= th]
    print("vocabulary size, occurance >= %d: %d" % (th, len(vocab_truncate)))
  return 

def get_vocab(training_set, word2id, id2word, vocab_size_threshold=3):
  """Get the vocabulary from the training set"""
  vocab = []
  for s in training_set:
    vocab.extend(s)

  vocab = Counter(vocab)
  print('%d words in total' % len(vocab))
  vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]

  i = len(word2id)
  for w in vocab_truncate:
    word2id[w] = i
    id2word[i] = w
    i += 1
  
  assert(len(word2id) == len(id2word))
  print("vocabulary size: %d" % len(word2id))
  return word2id, id2word