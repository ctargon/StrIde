#
# FILE: dataset.py
# USE:  Creates object that holds data and labels for gene specified dataset
#

import numpy as np
import random
import sys

class data_t(object):
	def __init__(self, data, labels=None, metas=None):
		self.labels = labels
		self.data = data
		self.metas = metas
		self.num_examples = data.shape[0]

	def next_batch(self, batch_size, index, metas=False):
		idx = index * batch_size
		n_idx = index * batch_size + batch_size
		if metas:
			return self.data[idx:n_idx, :], self.labels[idx:n_idx, :], self.metas[idx:n_idx, :]
		else:
			return self.data[idx:n_idx, :], self.labels[idx:n_idx, :]

	def permute(self, data, idxs):
		# data: [batch, 26, 26]
		out = []
		for i in range(data.shape[0]):
			buf = data[i, idxs]
			buf = buf[:, idxs]
			out.append(buf)
		return np.array(out)

# expects a numpy array of data and a corresponding numpy array of labels
# samples on the rows, features on the columns
class DataContainer:
	def __init__(self, data, labels, train_split=0.8, test_split=0.2):
		assert(data.shape[0] == labels.shape[0])
		self.num_classes = labels.shape[1]
		self.class_counts = {}
		self.train, self.test = self.partition(data, labels, train_split, test_split)


	#
	# USAGE:
	#	Shuffle training dataset
	def shuffle(self):
		idxs = np.arange(self.train.data.shape[0])
		np.random.shuffle(idxs)
		self.train.data = np.squeeze(self.train.data[idxs])
		self.train.labels = np.squeeze(self.train.labels[idxs])
		self.train.metas = np.squeeze(self.train.metas[idxs])

	#
	# USAGE:
	#	shuffle the data and labels in the same order for a data set and transform the data and labels into numpy arrays
	# PARAMS:
	#	data:	the data values for a dataset
	# 	labels: the labels associated with data for a dataset
	#
	def shuffle_and_transform(self, data, labels, metas):
		stacked_d = np.vstack(data)
		stacked_l = np.vstack(labels)
		stacked_metas = np.concatenate(metas)

		samples = random.sample(range(stacked_d.shape[0]),stacked_d.shape[0])

		# convert lists to numpy arrays
		stacked_d = stacked_d[samples]
		stacked_l = stacked_l[samples]
		stacked_metas = stacked_metas[samples]

		return data_t(stacked_d, stacked_l, stacked_metas)

	#
	# USAGE:
	#       partition dataset into train/test sets
	#
	def partition(self, data, labels, train_split=0.8, test_split=0.2):
		x_train = []
		y_train = []
		metas_train = []
		x_test = []
		y_test = []
		metas_test = []

		metas = np.bincount(np.where(data.any(axis=2))[0])

		for i in range(self.num_classes):
			# find where the labels are equal to the certain class
			idxs = np.where(np.argmax(labels, axis=1) == i)[0]
			np.random.shuffle(idxs)

			# record the class count information
			self.class_counts[str(i)] = idxs.shape[0]

			# get the int that splits the train/test sets
			split = int(train_split * idxs.shape[0])

			# append class data to respective lists
			x_train.append(data[idxs[:split]])
			y_train.append(labels[idxs[:split]])
			metas_train.append(metas[idxs[:split]])

			x_test.append(data[idxs[split:]])
			y_test.append(labels[idxs[split:]])
			metas_test.append(metas[idxs[split:]])

		# format into datacontainer 
		train = self.shuffle_and_transform(x_train, y_train, metas_train)
		test = self.shuffle_and_transform(x_test, y_test, metas_test)

		return [train, test]



class DataContainer_nolabel:
	def __init__(self, data, num_classes):
		self.num_classes = num_classes
		self.class_counts = {}
		self.test = self.partition(data)


	#
	# USAGE:
	#	TODO: What is this use @Colin
	def shuffle(self):
		idxs = np.arange(self.train.data.shape[0])
		np.random.shuffle(idxs)
		self.train.data = np.squeeze(self.train.data[idxs])
		self.train.labels = np.squeeze(self.train.labels[idxs])

	#
	# USAGE:
	#	shuffle the data and labels in the same order for a data set and transform the data and labels into numpy arrays
	# PARAMS:
	#	data:	the data values for a dataset
	# 	labels: the labels associated with data for a dataset
	#
	def transform(self, data):
		stacked_d = np.vstack(data)
		return data_t(stacked_d)

	def partition(self, data):
		x_test = []

		for i in range(self.num_classes):
			# find where the labels are equal to the certain class
			idxs = np.where(np.argmax(labels, axis=1) == i)[0]
			np.random.shuffle(idxs)

			# record the class count information
			self.class_counts[str(i)] = idxs.shape[0]

			# get the int that splits the train/test sets
			split = int(train_split * idxs.shape[0])

			# append class data to respective lists
			x_train.append(data[idxs[:split]])
			y_train.append(labels[idxs[:split]])

			x_test.append(data[idxs[split:]])
			y_test.append(labels[idxs[split:]])

		# format into datacontainer 
		train = self.shuffle_and_transform(x_train, y_train)
		test = self.shuffle_and_transform(x_test, y_test)

		return [train, test]




