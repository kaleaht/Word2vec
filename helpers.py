from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nltk.corpus import stopwords


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


def load_zip(file):
    with zipfile.ZipFile(file) as f:
        vocabulary = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print("Sanoja yhteensä koko datassa "+str(len(vocabulary)))

    # 1 pituiset pois
    vocabulary = list(filter(lambda word: len(word) > 1, vocabulary))
    
    print(len(vocabulary))
    
   # vocabulary = [word for word in vocabulary if word not in stopwords.words('finnish')]
    print(len(vocabulary))


    return vocabulary
