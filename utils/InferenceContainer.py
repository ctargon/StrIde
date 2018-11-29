#
# FILE: dataset.py
# USE:  Creates object that holds data and labels for gene specified dataset
#

import numpy as np
import random
import sys

class data_t(object):
	def __init__(self, data, labels):
		self.labels = labels
		self.data = data
		self.num_examples = data.shape[0]

	def next_batch(self, batch_size, index):
		idx = index * batch_size
		n_idx = index * batch_size + batch_size
		return self.data[idx:n_idx, :], self.labels[idx:n_idx, :]


# expects a numpy array of data and a corresponding numpy array of labels
# samples on the rows, features on the columns
class InferenceContainer:
	def __init__(self, data, labels):
		assert(data.shape[0] == labels.shape[0])
		self.num_classes = labels.shape[1]
		self.test = data_t(data, labels)



