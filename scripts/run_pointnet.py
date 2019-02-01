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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

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
									choices=['mlp', 'cnn', 'pc'], default='pc')
	parser.add_argument('--weights', help='folder to save weights to', type=str, required=True)
	parser.add_argument('--nepochs', help='number of epochs for training', type=int, required=False,default=100)
	parser.add_argument('--noise', help='type of noise to input', type=str, required=True, default='normal')
	parser.add_argument('--p1', help='param 1 for noise (mu if normal, lower bound if uniform)', type=float, required=True)
	parser.add_argument('--p2', help='param 2 for noise (sigma if normal, upper bound if uniform)', type=float, required=True)


	args = parser.parse_args()

	if not os.path.exists(args.weights):
		print('creating weight directory ' + str(args.weights))
		os.makedirs(args.weights)

	print('loading numpy data...')
	data = np.load(args.dataset)
	labels = np.load(args.labels)

	print('converting to DataContainer format...')
	dc = DC(data=data, labels=labels)

	if args.net == 'mlp':
		net = MLP(epochs=args.nepochs,
				  h_units=[512, 128, 32],
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)
	
	if args.net == 'cnn':
		net = CNN(epochs=args.nepochs,
				  batch_size=64,
				  n_input=dc.train.data.shape[-1],
				  verbose=1)

	if args.net == 'pc':
		net = PointNet(epochs=args.nepochs,
					   batch_size=64,
					   n_points=dc.train.data.shape[1],
					   n_classes=dc.train.labels.shape[-1],
					   n_input=dc.train.data.shape[-1], 
					   verbose=1,
					   save=1,
					   noise=args.noise,
					   params=[args.p1, args.p2],
                       weights_dir=args.weights)


	print('train shape: ' + str(dc.train.data.shape))
	print('label shape: ' + str(dc.train.labels.shape))
	print('test shape: ' + str(dc.test.data.shape))
	print('label shape: ' + str(dc.test.labels.shape))

	acc = net.run(dc)

	print('final accuracy: ' + str(acc))

