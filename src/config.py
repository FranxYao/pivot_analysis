"""The Pivot Analysis Configuration"""

class Config(object):
  dataset_name = 'yelp' # ['yelp', 'amazon', 'caption', 'gender', 'paper', 
                        # 'politics', 'reddit', 'twitter']

  model = 'cmu' # 'cmu', 'mit'

  show_statistics = True
  dataset_base_path = '../data/'
  output_path = '../outputs/'
  set_to_test = 'test'
  test_epoch = ''

  vocab_cnt_thres = 5
  vocab_size = -1
  pad = '_PAD'
  unk = '_UNK'

  is_bigram = False 
  is_trigram = False
  filter_stop_words = 0 # do not filter stop words in building vocabulary 

  pivot_thres_cnt = 1 # larger = higher confidence 
  prec_thres = 0.7 # larger = higher confidence 
  recl_thres = 0.0 # larger = higher confidence 

  classifier = 'none'  # 'cnn', 'fc' or 'none'
  max_training_case = 80000
  max_test_case = 10000
  num_epoch = 5
  batch_size = 200

  max_slen = {'yelp': 20,
              'amazon': 30,
              'caption': 30,
              'gender': 30,
              'paper': 30,
              'politics': 30,
              'reddit': 100,
              'twitter': 35}
  style2id = {'yelp':     {0: 'negative', 1: 'positive'},
              'amazon':   {0: 'negative', 1: 'positive'},
              'caption':  {0: 'humorous', 1: 'romantic'},
              'gender':   {0: 'male',     1: 'female'},
              'paper':    {0: 'academic', 1: 'journalism'},
              'politics': {0: 'democratic', 1: 'republican'},
              'reddit':   {0: 'impolite', 1: 'polite'},
              'twitter':  {0: 'impolite', 1: 'polite'}}
  id2style = {'yelp':     {'negative': 0, 'positive': 1},
              'amazon':   {'negative': 0, 'positive': 1},
              'caption':  {'humorous': 0, 'romantic': 1},
              'gender':   {'male':     0, 'female':   1},
              'paper':    {'academic': 0, 'journalism': 1},
              'politics': {'democratic': 0, 'republican': 1},
              'reddit':   {'impolite': 0, 'polite': 1},
              'twitter':  {'impolite': 0, 'polite': 1}}


  def parse_arg(self, args):
    print(args)
    # dataset 
    self.dataset_name = args.dataset
    self.model = args.model
    self.filter_stop_words = args.filter_stop_words
    self.set_to_test = args.set_to_test
    self.test_epoch = args.test_epoch
    if(args.bigram): self.is_bigram = True
    if(args.trigram): self.is_trigram = True 
    if(args.vocab_cnt_thres != -1): self.vocab_cnt_thres = args.vocab_cnt_thres
    if(args.pivot_thres_cnt != -1): self.pivot_thres_cnt = args.pivot_thres_cnt
    if(args.prec_thres != -1): self.prec_thres = args.prec_thres
    if(args.recl_thres != -1): self.recl_thres = args.recl_thres
    if(args.max_training_case != -1):
      self.max_training_case = args.max_training_case
    if(args.num_epoch != -1): self.num_epoch = args.num_epoch


  