#/usr/bin/python

'''
	This script can be used to run a specified dataset

'''

import numpy as np
import sys, argparse
import os
import json
import time

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from models.mlp import MLP
from models.cnn import CNN
from models.pointnet import PointNet
from utils.DataContainer import DataContainer as DC




if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset')
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
	parser.add_argument('--net', help='which type of network to run (mlp/cnn)', type=str, required=False, \
									choices=['mlp', 'cnn', 'pc'], default='mlp')
	parser.add_argument('--n_points', help='number of points to use from original dataset', type=int, required=False, default=25)

	args = parser.parse_args()

	print('loading numpy data...')
	data = np.load(args.dataset)
	labels = np.load(args.labels)

	print('converting to DataContainer format...')
	dc = DC(data=data, labels=labels)

	# trim distance matrices for experiments
	#dc.train.data = dc.train.data[:,:args.n_points,:]
	#dc.test.data = dc.test.data[:,:args.n_points,:]

	if args.net == 'mlp':
		# dc.train.data = dc.train.data.reshape(dc.train.data.shape[0], -1)
		# dc.test.data = dc.test.data.reshape(dc.test.data.shape[0], -1)
		# triu_i = np.triu_indices(dc.train.data.shape[-1], k=1)
		# dc.train.data = dc.train.data[:, triu_i[0], triu_i[1]]
		# dc.test.data = dc.test.data[:, triu_i[0], triu_i[1]]
		net = MLP(epochs=40,
				  h_units=[512, 128, 32],
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)
	
	if args.net == 'cnn':
		net = CNN(epochs=20,
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)

	if args.net == 'pc':
		net = PointNet(epochs=50,
					   batch_size=64,
					   n_points=dc.train.data.shape[1],
					   n_classes=dc.train.labels.shape[-1],
					   n_input=3, 
					   verbose=1,
					   save=1,
                       weights_file='/scratch3/ctargon/weights/r2.0/r2')


	print('train shape: ' + str(dc.train.data.shape))
	print('test shape: ' + str(dc.test.data.shape))

	acc = net.run(dc)

	print('final accuracy: ' + str(acc))

