"""Convert dataset format to different model requirement"""

import argparse

from config import Config
from data_utils import build_vocab

def add_arguments(config):
  parser = argparse.ArgumentParser(description='Command line arguments')

  parser.add_argument('--model', type=str, default='cmu',
                      help='The model')
  parser.add_argument('--dataset', type=str, default='amazon',
                      help='The model')
  return parser.parse_args()

def main():
  config = Config()
  args = add_arguments(config)

  # read sentences
  data_path = config.dataset_base_path + args.dataset + '/'

  train_neg = open(data_path + 'train.0').readlines()
  train_pos = open(data_path + 'train.1').readlines()
  train_labels = [0] * len(train_neg) + [1] * len(train_pos)
  train = train_neg + train_pos
  train = [s.split() for s in train]
  train = [s[:30] if len(s) > 30 else s for s in train]

  dev_neg = open(data_path + 'dev.0').readlines()
  dev_pos = open(data_path + 'dev.1').readlines()
  dev_labels = [0] * len(dev_neg) + [1] * len(dev_pos)
  dev = dev_neg + dev_pos
  dev = [s.split() for s in dev]
  dev = [s[:30] if len(s) > 30 else s for s in dev]

  test_neg = open(data_path + 'test.0').readlines()
  test_pos = open(data_path + 'test.1').readlines()
  test_labels = [0] * len(test_neg) + [1] * len(test_pos)
  test = test_neg + test_pos
  test = [s.split() for s in test]
  test = [s[:30] if len(s) > 30 else s for s in test]

  word2id, _, _, _, _, _ = build_vocab(train, filter_stop_words=0)
  if(args.model == 'cmu'):
    with open(args.dataset + '.train.text', 'w') as fd:
      for l in train: fd.write(' '.join(l) + '\n')
    with open(args.dataset + '.train.labels', 'w') as fd:
      for l in train_labels: fd.write(str(l) + '\n')

    with open(args.dataset + '.dev.text', 'w') as fd:
      for l in dev: fd.write(' '.join(l) + '\n')
    with open(args.dataset + '.dev.labels', 'w') as fd:
      for l in dev_labels: fd.write(str(l) + '\n')

    with open(args.dataset + '.test.text', 'w') as fd:
      for l in test: fd.write(' '.join(l) + '\n')
    with open(args.dataset + '.test.labels', 'w') as fd:
      for l in test_labels: fd.write(str(l) + '\n')

    with open('vocab', 'w') as fd:
      for w in word2id: fd.write('%s\n' % w)
  return 

if __name__ == '__main__':
  main()