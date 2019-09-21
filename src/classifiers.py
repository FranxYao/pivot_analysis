"""The Logistic Baseline Calssifier for Pivot Analysis

Yao Fu, Columbia University 
yao.fu@columbia.edu
Sat Jun 22nd 2019 
"""

import numpy as np 
import keras.layers as layers

from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding
from keras.layers import Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, concatenate
from keras import optimizers
from keras.models import Model


class LogisticClassifier(object):
  """The logistic classifier, the baseline model"""

  def __init__(self, config):
    self.model = LogisticRegression(solver='lbfgs')

    self.max_training_case = config.max_training_case
    return 

  def train(self, train_data, labels):
    if(len(train_data) >= self.max_training_case): 
      sample_id = np.random.choice(
        len(train_data), self.max_training_case, False)
      train_data = train_data[sample_id]
      train_labels = train_labels[sample_id]
    self.model.fit(train_data, labels)
    return 

  def test(self, test_data, labels):
    pred = self.model.predict(test_data)
    acc = np.sum(pred == labels) / float(len(labels))
    print('accuracy: %.4f' % acc)
    return 

  def word_saliency(self):
    """Word saliency analysis, output strong words, """
    return 

class CNNClassifier(object):
  def __init__(self, config):
    """
    Convolution neural network model for sentence classification.
    Parameters
    Sentence CNN by Y.Kim
    ----------
    EMBEDDING_DIM: Dimension of the embedding space.
    MAX_SEQUENCE_LENGTH: Maximum length of the sentence.
    MAX_NB_WORDS: Maximum number of words in the vocabulary.
    embeddings_index: A dict containing words and their embeddings.
    word_index: A dict containing words and their indices.
    labels_index: A dict containing the labels and their indices.
    Returns
    -------
    compiled keras model
    """
    self.batch_size = config.batch_size
    self.num_epoch = config.num_epoch

    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = config.max_slen[config.dataset_name]
    # embedding_matrix = np.zeros((config.vocab_size, EMBEDDING_DIM))
    embedding_layer = Embedding(config.vocab_size,
      EMBEDDING_DIM,
      input_length=MAX_SEQUENCE_LENGTH,
      trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # add first conv filter
    embedded_sequences = Reshape(
      (MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)

    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)

    # add second conv filter.
    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)

    # add third conv filter.
    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)

    # concate the conv layers
    alpha = concatenate([x,y,z])
    # flatted the pooled features.
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)
    # predictions
    preds = Dense(1, activation='sigmoid')(alpha)

    # build model
    model = Model(sequence_input, preds)
    opt = optimizers.Adam(lr=0.0001)
        
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    self.model = model
    return

  def train(self, train_data, train_labels):
    history = self.model.fit(train_data, train_labels, batch_size=self.batch_size,
    epochs=self.num_epoch, verbose=1, validation_split=0.1)
    return

  def test(self, test_data, labels):
    score = self.model.evaluate(test_data, labels, verbose=1)
    acc = score[1]
    print('accuracy: %.4f' % acc)
    return

class FFClassifier(object):
  """The CNN classifier"""

  def __init__(self, config):
    self.num_epoch = config.num_epoch
    self.batch_size = config.batch_size
    self.max_training_case = config.max_training_case

    model = Sequential()
    # model.add(Dense(200, input_shape=(config.vocab_size, )))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, input_shape=(config.vocab_size, )))
    model.add(Activation('sigmoid'))
    model.compile(
      loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    self.model = model
    return 

  def train(self, train_data, train_labels):
    if(len(train_data) >= self.max_training_case): 
      sample_id = np.random.choice(
        len(train_data), self.max_training_case, False)
      train_data = train_data[sample_id]
      train_labels = train_labels[sample_id]
    history = self.model.fit(train_data, train_labels, batch_size=self.batch_size, 
      epochs=self.num_epoch, verbose=1, validation_split=0.1)
    return 
  
  def test(self, test_data, labels):
    score = self.model.evaluate(test_data, labels, verbose=1)
    acc = score[1]
    print('accuracy: %.4f' % acc)
    return 


